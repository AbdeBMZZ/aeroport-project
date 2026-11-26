/*
 * Informatique Temps-Réel – Projet : l'aéroport à piste unique.
 *
 * Auteurs : Abdellah Boumaiza et Estebane Vanduynslager
 *
 * Contrôleur de l'aéroport. Trois threads (inspection, atterrissage,
 * déneigement) se partagent la piste selon un ordonnancement EDF calculé à la
 * main sur une hyperpériode (voir le rapport) :
 *
 *     pas   :  0  1  2  3  4  5  6  7  8  9 10 11
 *     tâche :  I  A  A  N  I  N  N  A  A  I  -  -
 *
 * Chaque thread se réveille au début de chacun de ses pas avec un réveil absolu
 * (clock_nanosleep + TIMER_ABSTIME sur CLOCK_REALTIME) calculé depuis un instant
 * de référence commun t0. Aucune primitive de synchronisation n'est utilisée
 * pour imposer l'ordonnancement : les délais suffisent.
 *
 * Chaque pas d'atterrissage est surveillé par un watchdog (timer SIGEV_THREAD)
 * qui appelle abort_landing avant la fin du créneau si le pas déborde.
 */

#include "airport.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- Ordonnancement ---------------------------------------------------------

// Pour chaque instance de la tâche dans l'hyperpériode, décalage (en pas) de
// chacun de ses pas par rapport à l'activation de l'instance.

// Inspection : période 4, 1 pas.
static const unsigned int inspection_offsets[][INSPECTION_STEPS] = {
    {0}, // I1 : activation 0, pas 0
    {0}, // I2 : activation 4, pas 4
    {1}, // I3 : activation 8, pas 9
};

// Atterrissage : période 6, 2 pas.
static const unsigned int landing_offsets[][LANDING_STEPS] = {
    {1, 2}, // A1 : activation 0, pas 1 et 2
    {1, 2}, // A2 : activation 6, pas 7 et 8
};

// Déneigement : période 12, 3 pas.
static const unsigned int plow_offsets[][PLOW_STEPS] = {
    {3, 5, 6}, // N1 : activation 0, pas 3, 5 et 6
};

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define INSPECTION_INSTANCES COUNT_OF(inspection_offsets)
#define LANDING_INSTANCES COUNT_OF(landing_offsets)
#define PLOW_INSTANCES COUNT_OF(plow_offsets)
#define HYPERPERIOD (INSPECTION_INSTANCES * INSPECTION_PERIOD)

_Static_assert(INSPECTION_INSTANCES * INSPECTION_PERIOD == LANDING_INSTANCES * LANDING_PERIOD,
               "the inspection and landing tables must cover the same hyperperiod");
_Static_assert(INSPECTION_INSTANCES * INSPECTION_PERIOD == PLOW_INSTANCES * PLOW_PERIOD,
               "the inspection and plowing tables must cover the same hyperperiod");
_Static_assert(JITTER >= 2, "the watchdog needs a margin of JITTER / 2 ms before the end of the slot");

// --- Temps ------------------------------------------------------------------

#define NS_PER_S 1000000000LL
#define NS_PER_MS 1000000LL

// Durée d'un pas sur la piste (ms) et instant de déclenchement du watchdog,
// relatif au début du créneau (ms). Le watchdog libère la piste JITTER / 2 ms
// avant le créneau suivant, et ne gêne pas un pas normal démarré avec moins de
// JITTER / 2 ms de retard.
#define RUNWAY_TIME_MS ((long long)(STEP_TIME - JITTER))
#define WATCHDOG_DELAY_MS ((long long)(STEP_TIME - JITTER / 2))

// Signal utilisé uniquement pour réveiller les threads à la fermeture.
#define WAKEUP_SIGNAL SIGUSR2

#define NUM_THREADS 3

// --- État partagé -----------------------------------------------------------

static airport_t *airport;

// Instant de référence commun, fixé juste après open_airport et avant la
// création des threads (pthread_create garantit sa visibilité).
static struct timespec t0;

// Nombre total d'avions arrivés (écrit uniquement par le gestionnaire de
// signal) et nombre d'avions posés (écrit uniquement par le thread
// d'atterrissage).
static volatile sig_atomic_t planes_arrived = 0;
static volatile sig_atomic_t planes_landed = 0;

// Demande d'arrêt des threads (fermeture de l'aéroport ou erreur).
static atomic_int stop_requested = 0;

// Instant (ns, CLOCK_REALTIME) où le watchdog du pas d'atterrissage en cours
// doit se déclencher, 0 si aucun pas n'est surveillé.
static atomic_llong watchdog_deadline_ns = 0;

// --- Fonctions utilitaires --------------------------------------------------

static long long timespec_to_ns(const struct timespec *t) {
    return (long long)t->tv_sec * NS_PER_S + t->tv_nsec;
}

static struct timespec ns_to_timespec(long long ns) {
    struct timespec t;
    t.tv_sec = (time_t)(ns / NS_PER_S);
    t.tv_nsec = (long)(ns % NS_PER_S);
    return t;
}

static long long now_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return timespec_to_ns(&now);
}

// Début absolu (ns) du pas numéro `slot`, compté depuis t0.
static long long slot_start_ns(unsigned long long slot) {
    return timespec_to_ns(&t0) + (long long)(slot * STEP_TIME) * NS_PER_MS;
}

// Temps écoulé depuis t0 (s), pour la trace.
static double elapsed_s(void) {
    return (double)(now_ns() - timespec_to_ns(&t0)) / 1e9;
}

// Dort jusqu'à l'instant absolu `deadline` (CLOCK_REALTIME).
// Retourne 0 à l'instant voulu, -1 si l'arrêt est demandé ou en cas d'erreur.
static int delay_until(long long deadline) {
    struct timespec ts = ns_to_timespec(deadline);
    for (;;) {
        if (atomic_load(&stop_requested)) {
            return -1;
        }
        int err = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts, NULL);
        if (err == 0) {
            return atomic_load(&stop_requested) ? -1 : 0;
        }
        if (err != EINTR) {
            fprintf(stderr, "clock_nanosleep: %s\n", strerror(err));
            return -1;
        }
        // EINTR : réveil par un signal, on se rendort (même instant absolu).
    }
}

static const char *error_name(error_t e) {
    switch (e) {
    case OK: return "OK";
    case AIRPORT_CLOSED: return "AIRPORT_CLOSED";
    case COLLISION: return "COLLISION";
    case INVALID_STATE: return "INVALID_STATE";
    case NULL_PTR: return "NULL_PTR";
    case MALLOC: return "MALLOC";
    case TIMER_CREATE: return "TIMER_CREATE";
    case TIMER_SETTIME: return "TIMER_SETTIME";
    case TIMER_DELETE: return "TIMER_DELETE";
    case MUTEX: return "MUTEX";
    case LANDING_ABORTED: return "LANDING_ABORTED";
    case NO_PLANE: return "NO_PLANE";
    case NO_SIGNAL_HANDLER: return "NO_SIGNAL_HANDLER";
    }
    return "unknown error";
}

// Message d'arrêt d'un thread après une erreur de la librairie.
static void report_stop(const char *task, error_t e) {
    if (e == AIRPORT_CLOSED) {
        printf("[%7.3f s] %s thread: airport closed, stopping.\n", elapsed_s(), task);
    } else {
        printf("[%7.3f s] %s thread: error %s, stopping.\n", elapsed_s(), task, error_name(e));
    }
    fflush(stdout);
}

// --- Gestionnaires de signaux -----------------------------------------------

// Appelé à chaque arrivée d'avion (ARRIVAL_SIGNAL). Le plus court possible :
// on compte l'avion, le thread d'atterrissage fera le reste.
static void on_plane(int sig) {
    (void)sig;
    planes_arrived++;
}

// Ne fait rien : sert uniquement à interrompre clock_nanosleep (EINTR).
static void on_wakeup(int sig) {
    (void)sig;
}

// --- Tâche d'inspection -----------------------------------------------------

static void *inspection_thread(void *arg) {
    (void)arg;
    for (unsigned long long k = 0; k * INSPECTION_PERIOD < OPENING_STEPS; ++k) {
        unsigned long long activation = k * INSPECTION_PERIOD;
        const unsigned int *offsets = inspection_offsets[k % INSPECTION_INSTANCES];
        for (unsigned int s = 0; s < INSPECTION_STEPS; ++s) {
            unsigned long long slot = activation + offsets[s];
            long long start = slot_start_ns(slot);
            if (delay_until(start) != 0) {
                return NULL;
            }
            // Trop en retard : le pas déborderait sur le créneau suivant.
            if (now_ns() - start > (long long)JITTER * NS_PER_MS) {
                printf("[%7.3f s] Inspection: woke up too late for slot %llu, step skipped\n",
                       elapsed_s(), slot);
                fflush(stdout);
                continue;
            }
            error_t res = inspect_runway(airport);
            if (res != OK) {
                report_stop("Inspection", res);
                return NULL;
            }
        }
    }
    return NULL;
}

// --- Tâche d'atterrissage ---------------------------------------------------

// Notification du watchdog (thread créé par le timer). N'interrompt le pas
// que si un pas est surveillé et que son échéance est atteinte : une
// notification en retard, venant d'un pas précédent, est ignorée.
static void watchdog_handler(union sigval sv) {
    (void)sv;
    long long deadline = atomic_load(&watchdog_deadline_ns);
    if (deadline == 0 || now_ns() < deadline) {
        return;
    }
    abort_landing(airport);
}

static int arm_watchdog(timer_t watchdog, long long deadline) {
    struct itimerspec its = {0};
    its.it_value = ns_to_timespec(deadline);
    atomic_store(&watchdog_deadline_ns, deadline);
    if (timer_settime(watchdog, TIMER_ABSTIME, &its, NULL) != 0) {
        atomic_store(&watchdog_deadline_ns, 0);
        return -1;
    }
    return 0;
}

static void disarm_watchdog(timer_t watchdog) {
    struct itimerspec its = {0};
    atomic_store(&watchdog_deadline_ns, 0);
    timer_settime(watchdog, 0, &its, NULL);
}

static void *landing_thread(void *arg) {
    (void)arg;

    timer_t watchdog;
    struct sigevent sev = {0};
    sev.sigev_notify = SIGEV_THREAD;
    sev.sigev_notify_function = watchdog_handler;
    sev.sigev_notify_attributes = NULL;
    if (timer_create(CLOCK_REALTIME, &sev, &watchdog) != 0) {
        perror("timer_create (watchdog)");
        return NULL;
    }

    unsigned int progress = 0; // pas réussis de l'atterrissage en cours

    for (unsigned long long k = 0; k * LANDING_PERIOD < OPENING_STEPS; ++k) {
        unsigned long long activation = k * LANDING_PERIOD;
        const unsigned int *offsets = landing_offsets[k % LANDING_INSTANCES];
        for (unsigned int s = 0; s < LANDING_STEPS; ++s) {
            unsigned long long slot = activation + offsets[s];
            long long start = slot_start_ns(slot);
            if (delay_until(start) != 0) {
                goto out;
            }
            // Aucun avion en attente : le créneau reste vide.
            if (planes_arrived == planes_landed) {
                printf("[%7.3f s] Landing slot %llu: no plane waiting\n", elapsed_s(), slot);
                fflush(stdout);
                continue;
            }
            // Le watchdog libère la piste avant le début du créneau suivant.
            long long watchdog_at = start + WATCHDOG_DELAY_MS * NS_PER_MS;
            // Trop en retard : le pas serait interrompu même sans débordement.
            if (now_ns() + RUNWAY_TIME_MS * NS_PER_MS > watchdog_at) {
                printf("[%7.3f s] Landing: woke up too late for slot %llu, step skipped\n",
                       elapsed_s(), slot);
                fflush(stdout);
                continue;
            }
            // Jamais de pas d'atterrissage sans watchdog.
            if (arm_watchdog(watchdog, watchdog_at) != 0) {
                perror("timer_settime (watchdog)");
                continue;
            }
            error_t res = land_step(airport);
            disarm_watchdog(watchdog); // avant tout autre traitement
            switch (res) {
            case OK:
                if (++progress == LANDING_STEPS) {
                    progress = 0;
                    planes_landed++;
                }
                break;
            case LANDING_ABORTED:
                // Remise des gaz : l'atterrissage recommence depuis le début.
                progress = 0;
                printf("[%7.3f s] Watchdog: landing step aborted in slot %llu, landing restarts\n",
                       elapsed_s(), slot);
                fflush(stdout);
                break;
            case NO_PLANE:
                // Ne devrait pas arriver : la librairie compte l'avion avant
                // d'envoyer le signal.
                printf("[%7.3f s] Landing slot %llu: library reports no plane\n", elapsed_s(), slot);
                fflush(stdout);
                break;
            default:
                report_stop("Landing", res);
                goto out;
            }
        }
    }

out:
    disarm_watchdog(watchdog);
    timer_delete(watchdog);
    return NULL;
}

// --- Tâche de déneigement ---------------------------------------------------

static void *plow_thread(void *arg) {
    (void)arg;
    for (unsigned long long k = 0; k * PLOW_PERIOD < OPENING_STEPS; ++k) {
        unsigned long long activation = k * PLOW_PERIOD;
        const unsigned int *offsets = plow_offsets[k % PLOW_INSTANCES];
        for (unsigned int s = 0; s < PLOW_STEPS; ++s) {
            unsigned long long slot = activation + offsets[s];
            long long start = slot_start_ns(slot);
            if (delay_until(start) != 0) {
                return NULL;
            }
            // Trop en retard : le pas déborderait sur le créneau suivant.
            if (now_ns() - start > (long long)JITTER * NS_PER_MS) {
                printf("[%7.3f s] Plowing: woke up too late for slot %llu, step skipped\n",
                       elapsed_s(), slot);
                fflush(stdout);
                continue;
            }
            error_t res = plow_step(airport);
            if (res != OK) {
                report_stop("Plowing", res);
                return NULL;
            }
        }
    }
    return NULL;
}

// --- Vérification des tables ------------------------------------------------

// Place les pas d'une instance dans la table d'occupation de l'hyperpériode.
// Vérifie qu'ils sont dans l'ordre, avant l'échéance et sur une piste libre.
static int place_steps(char *runway, char name, unsigned int instance, unsigned int period,
                       const unsigned int *offsets, unsigned int steps) {
    for (unsigned int s = 0; s < steps; ++s) {
        if (offsets[s] >= period || (s > 0 && offsets[s] <= offsets[s - 1])) {
            fprintf(stderr, "Schedule error: step %u of %c%u is misplaced\n", s, name, instance + 1);
            return 0;
        }
        unsigned int slot = instance * period + offsets[s];
        if (runway[slot] != '-') {
            fprintf(stderr, "Schedule error: %c%u and %c share slot %u\n",
                    name, instance + 1, runway[slot], slot);
            return 0;
        }
        runway[slot] = name;
    }
    return 1;
}

// Vérifie les tables de décalages et affiche l'ordonnancement.
static int check_schedule(void) {
    char runway[HYPERPERIOD];
    memset(runway, '-', sizeof(runway));
    for (unsigned int i = 0; i < INSPECTION_INSTANCES; ++i) {
        if (!place_steps(runway, 'I', i, INSPECTION_PERIOD, inspection_offsets[i], INSPECTION_STEPS)) {
            return 0;
        }
    }
    for (unsigned int i = 0; i < LANDING_INSTANCES; ++i) {
        if (!place_steps(runway, 'A', i, LANDING_PERIOD, landing_offsets[i], LANDING_STEPS)) {
            return 0;
        }
    }
    for (unsigned int i = 0; i < PLOW_INSTANCES; ++i) {
        if (!place_steps(runway, 'N', i, PLOW_PERIOD, plow_offsets[i], PLOW_STEPS)) {
            return 0;
        }
    }
    printf("Schedule over a hyperperiod of %u slots:", (unsigned int)HYPERPERIOD);
    for (unsigned int slot = 0; slot < HYPERPERIOD; ++slot) {
        printf(" %c", runway[slot]);
    }
    printf("\n");
    fflush(stdout);
    return 1;
}

// --- Thread principal -------------------------------------------------------

// Demande l'arrêt des threads, les réveille et attend leur fin.
static void stop_threads(pthread_t *threads, int count) {
    atomic_store(&stop_requested, 1);
    for (int i = 0; i < count; ++i) {
        pthread_kill(threads[i], WAKEUP_SIGNAL);
    }
    for (int i = 0; i < count; ++i) {
        int err = pthread_join(threads[i], NULL);
        if (err != 0) {
            fprintf(stderr, "pthread_join: %s\n", strerror(err));
        }
    }
}

int main(void) {
    if (!check_schedule()) {
        return EXIT_FAILURE;
    }

    // 1. Initialisation de l'aéroport et des gestionnaires de signaux.
    error_t res = init_airport(&airport);
    if (res != OK) {
        fprintf(stderr, "init_airport: %s\n", error_name(res));
        return EXIT_FAILURE;
    }

    struct sigaction sa = {0};
    sa.sa_handler = on_plane;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(ARRIVAL_SIGNAL, &sa, NULL) != 0) {
        perror("sigaction (ARRIVAL_SIGNAL)");
        free_airport(airport);
        return EXIT_FAILURE;
    }

    struct sigaction wake = {0};
    wake.sa_handler = on_wakeup;
    sigemptyset(&wake.sa_mask);
    wake.sa_flags = 0; // pas de SA_RESTART : interrompt clock_nanosleep
    if (sigaction(WAKEUP_SIGNAL, &wake, NULL) != 0) {
        perror("sigaction (WAKEUP_SIGNAL)");
        free_airport(airport);
        return EXIT_FAILURE;
    }

    // Ouverture : début des périodes de toutes les tâches.
    res = open_airport(airport);
    if (res != OK) {
        fprintf(stderr, "open_airport: %s\n", error_name(res));
        free_airport(airport);
        return EXIT_FAILURE;
    }

    // 2. Instant de référence commun, juste après l'ouverture.
    if (clock_gettime(CLOCK_REALTIME, &t0) != 0) {
        perror("clock_gettime");
        free_airport(airport);
        return EXIT_FAILURE;
    }

    // 3. Un thread par tâche. ARRIVAL_SIGNAL est bloqué dans ces threads (le
    // masque est hérité) : il est traité par le thread principal et
    // n'interrompt jamais leurs réveils.
    sigset_t arrival, old_mask;
    sigemptyset(&arrival);
    sigaddset(&arrival, ARRIVAL_SIGNAL);
    pthread_sigmask(SIG_BLOCK, &arrival, &old_mask);

    void *(*const functions[NUM_THREADS])(void *) = {inspection_thread, landing_thread, plow_thread};
    const char *names[NUM_THREADS] = {"inspection", "landing", "plowing"};
    pthread_t threads[NUM_THREADS];
    int created = 0;
    for (; created < NUM_THREADS; ++created) {
        int err = pthread_create(&threads[created], NULL, functions[created], NULL);
        if (err != 0) {
            fprintf(stderr, "pthread_create (%s): %s\n", names[created], strerror(err));
            break;
        }
    }

    pthread_sigmask(SIG_SETMASK, &old_mask, NULL);

    if (created < NUM_THREADS) {
        stop_threads(threads, created);
        free_airport(airport);
        return EXIT_FAILURE;
    }

    // 5. Attente de la fermeture, arrêt des threads, libération.
    outcome_t outcome = STILL_OPEN;
    res = wait_airport_closed(airport, &outcome);
    if (res != OK) {
        fprintf(stderr, "wait_airport_closed: %s\n", error_name(res));
    }

    // Les threads encore endormis (fermeture anticipée, ou thread
    // d'atterrissage sans avion) sont réveillés et s'arrêtent.
    stop_threads(threads, NUM_THREADS);

    switch (outcome) {
    case DAY_COMPLETED:
        printf("The day is over.\n");
        break;
    case TOO_MANY_INCIDENTS:
        printf("The airport closed early: too many incidents.\n");
        break;
    case CRASHED:
        printf("The airport closed early: crash on the runway.\n");
        break;
    default:
        printf("The airport closed in an unexpected state.\n");
        break;
    }
    printf("Controller: %d plane(s) arrived, %d landed.\n", (int)planes_arrived, (int)planes_landed);

    res = free_airport(airport);
    if (res != OK) {
        fprintf(stderr, "free_airport: %s\n", error_name(res));
        return EXIT_FAILURE;
    }
    return outcome == DAY_COMPLETED ? EXIT_SUCCESS : EXIT_FAILURE;
}
