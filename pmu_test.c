/*
 * pmu_test.c — PMU counter verification test for EIC7700X (ESWIN 770x)
 *
 * Tests the pmu_module kernel module in two layers:
 *
 *   Legacy path (IOCTL_PMU_SETUP):
 *     cycle, instret, and 3 perf-based HPM events (branch_miss, cache_miss,
 *     cache_ref).  These use the Linux perf subsystem and the EFI DTB.
 *
 *   Raw scan path (IOCTL_PMU_RAW_SCAN + IOCTL_PMU_OPEN_RAW):
 *     Probes all 512 class/mask combinations via SBI CFG_MATCH with raw event
 *     type, which writes mhpmeventX directly without a DTB lookup.  For every
 *     discovered event this test checks:
 *       (a) U-mode CSR accessibility — does csrr raise SIGILL?
 *       (b) Counter liveness — does it increment with any of three workloads?
 *
 * Build:
 *   gcc -O2 -o pmu_test pmu_test.c
 *
 * Run:
 *   sudo insmod pmu_module.ko
 *   sudo ./pmu_test
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/ioctl.h>

/* =========================================================
 * IOCTL interface — must match pmu_module.c exactly
 * ========================================================= */
struct pmu_setup_info {
    int      cpu;
    uint64_t scounteren_val;
    int      cycle_ok;
    int      instret_ok;
    int      branch_miss_ok;
    int      cache_miss_ok;
    int      cache_ref_ok;
    int      branch_miss_idx;
    int      cache_miss_idx;
    int      cache_ref_idx;
    uint64_t cycle_initial;
    uint64_t instret_initial;
    int      counters_running;
};

#define PMU_RAW_MAX_EVENTS  512   /* scan table capacity — must match pmu_module.c */
#define PMU_RAW_CTR_MAX       4   /* hardware HPM counters: hpm3–hpm6 */

struct pmu_raw_event {
    uint64_t event_data;
    int      ctr_idx;
};

struct pmu_raw_scan_result {
    int                  count;
    struct pmu_raw_event events[PMU_RAW_MAX_EVENTS];
};

struct pmu_open_raw_request {
    int      cpu;
    int      count;
    uint64_t event_data[PMU_RAW_MAX_EVENTS];
    int      ctr_idx[PMU_RAW_MAX_EVENTS];
    int      ok[PMU_RAW_MAX_EVENTS];
};

#define IOCTL_PMU_SETUP     _IOWR('p', 5, struct pmu_setup_info)
#define IOCTL_PMU_TEARDOWN  _IO  ('p', 6)
#define IOCTL_PMU_RAW_SCAN  _IOR ('p', 7, struct pmu_raw_scan_result)
#define IOCTL_PMU_OPEN_RAW  _IOWR('p', 8, struct pmu_open_raw_request)
#define IOCTL_PMU_CLOSE_RAW _IO  ('p', 9)

/* =========================================================
 * CSR read — covers all hpmcounter3..31
 * ========================================================= */
static inline uint64_t read_hpmcounter(int idx)
{
    uint64_t v = 0;
    switch (idx) {
    case  3: asm volatile("csrr %0,0xc03":"=r"(v)); break;
    case  4: asm volatile("csrr %0,0xc04":"=r"(v)); break;
    case  5: asm volatile("csrr %0,0xc05":"=r"(v)); break;
    case  6: asm volatile("csrr %0,0xc06":"=r"(v)); break;
    case  7: asm volatile("csrr %0,0xc07":"=r"(v)); break;
    case  8: asm volatile("csrr %0,0xc08":"=r"(v)); break;
    case  9: asm volatile("csrr %0,0xc09":"=r"(v)); break;
    case 10: asm volatile("csrr %0,0xc0a":"=r"(v)); break;
    case 11: asm volatile("csrr %0,0xc0b":"=r"(v)); break;
    case 12: asm volatile("csrr %0,0xc0c":"=r"(v)); break;
    case 13: asm volatile("csrr %0,0xc0d":"=r"(v)); break;
    case 14: asm volatile("csrr %0,0xc0e":"=r"(v)); break;
    case 15: asm volatile("csrr %0,0xc0f":"=r"(v)); break;
    case 16: asm volatile("csrr %0,0xc10":"=r"(v)); break;
    case 17: asm volatile("csrr %0,0xc11":"=r"(v)); break;
    case 18: asm volatile("csrr %0,0xc12":"=r"(v)); break;
    case 19: asm volatile("csrr %0,0xc13":"=r"(v)); break;
    case 20: asm volatile("csrr %0,0xc14":"=r"(v)); break;
    case 21: asm volatile("csrr %0,0xc15":"=r"(v)); break;
    case 22: asm volatile("csrr %0,0xc16":"=r"(v)); break;
    case 23: asm volatile("csrr %0,0xc17":"=r"(v)); break;
    case 24: asm volatile("csrr %0,0xc18":"=r"(v)); break;
    case 25: asm volatile("csrr %0,0xc19":"=r"(v)); break;
    case 26: asm volatile("csrr %0,0xc1a":"=r"(v)); break;
    case 27: asm volatile("csrr %0,0xc1b":"=r"(v)); break;
    case 28: asm volatile("csrr %0,0xc1c":"=r"(v)); break;
    case 29: asm volatile("csrr %0,0xc1d":"=r"(v)); break;
    case 30: asm volatile("csrr %0,0xc1e":"=r"(v)); break;
    case 31: asm volatile("csrr %0,0xc1f":"=r"(v)); break;
    default: break;
    }
    return v;
}

/* =========================================================
 * SIGILL-guarded CSR probe
 *
 * Attempts a csrr for the given counter index.  Returns 1 if
 * successful (counter is readable from U-mode), 0 if SIGILL
 * (scounteren bit not set, or counter does not exist).
 * ========================================================= */
static sigjmp_buf   g_probe_env;
static volatile int g_probe_sigill;

static void sigill_handler(int sig) { g_probe_sigill = 1; siglongjmp(g_probe_env, 1); }

/* Returns 1 = readable, 0 = SIGILL */
static int probe_csr_readable(int idx, uint64_t *val)
{
    uint64_t v = 0;
    g_probe_sigill = 0;
    signal(SIGILL, sigill_handler);
    if (sigsetjmp(g_probe_env, 1) == 0) {
        v = read_hpmcounter(idx);
    }
    signal(SIGILL, SIG_DFL);
    *val = v;
    return !g_probe_sigill;
}

/* =========================================================
 * Helpers
 * ========================================================= */
static void pin_to_cpu(int cpu)
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(cpu, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0)
        perror("  sched_setaffinity (non-fatal)");
}

static void sep(void) { puts("  -----------------------------------------------------------------------"); }
static const char *tick(int v) { return v ? "✓" : "✗ FAIL"; }

/* =========================================================
 * Workloads for liveness testing
 * =========================================================
 * Three workloads cover the major counter classes:
 *
 *   nop_sled   — advances cycle, instret, front-end events
 *   branch_loop — LFSR-driven unpredictable branches → branch_miss events
 *   cache_load  — strided accesses to a cold buffer → cache_miss/ref events
 * ========================================================= */
#define LIVENESS_ITERS  2000

static void workload_nop(void)
{
    for (int i = 0; i < LIVENESS_ITERS; i++)
        asm volatile("add x0,x0,x0" :::);
}

static void workload_branch(void)
{
    volatile uint32_t lfsr = 0xACE1u;
    for (int i = 0; i < LIVENESS_ITERS; i++) {
        uint32_t bit = ((lfsr>>0)^(lfsr>>2)^(lfsr>>3)^(lfsr>>5)) & 1;
        lfsr = (lfsr >> 1) | (bit << 15);
        if (lfsr & 1) lfsr ^= 0x1337;
    }
}

static void workload_cache(void)
{
    static volatile char buf[64 * 512];
    static int warm = 0;
    if (!warm) { for (int i = 0; i < (int)sizeof(buf); i++) buf[i] = (char)i; warm = 1; }
    volatile char sink = 0;
    for (int i = 0; i < 512; i++) sink ^= buf[i * 64];
    (void)sink;
}

/* Small trampoline targets at file scope (avoids nested-function definitions) */
static void tgt0(void) { asm volatile("nop"); }
static void tgt1(void) { asm volatile("nop"); }
static void tgt2(void) { asm volatile("nop"); }
static void tgt3(void) { asm volatile("nop"); }

static void ret_t0(void) { asm volatile("ret"); }
static void ret_t1(void) { asm volatile("ret"); }

/* Targeted workloads to provoke branch-target and pipeline mispredictions */
static void workload_indirect_branch(void)
{
    typedef void (*fn_t)(void);
    fn_t tbl[4] = { tgt0, tgt1, tgt2, tgt3 };

    uint32_t lfsr = 0x1234567;
    for (int i = 0; i < LIVENESS_ITERS; i++) {
        /* simple xorshift-ish LFSR to produce unpredictable targets */
        lfsr ^= lfsr << 13;
        lfsr ^= lfsr >> 17;
        lfsr ^= lfsr << 5;
        int idx = lfsr & 3;
        tbl[idx]();
    }
}

static void workload_return_mispredict(void)
{
    typedef void (*fn_t)(void);
    fn_t tbl[2] = { ret_t0, ret_t1 };

    uint32_t lfsr = 0xBEEF;
    for (int i = 0; i < LIVENESS_ITERS; i++) {
        lfsr = (lfsr >> 1) | (((lfsr ^ (lfsr << 3)) & 1) << 15);
        int idx = lfsr & 1;
        tbl[idx]();
    }
}

/* Run one liveness measurement: snapshot → workload → snapshot → delta */
static uint64_t measure_delta(int ctr_idx, void (*workload)(void))
{
    uint64_t before = read_hpmcounter(ctr_idx);
    workload();
    uint64_t after  = read_hpmcounter(ctr_idx);
    return after - before;
}

/* =========================================================
 * Step 5 per-instruction runners
 *
 * Each function executes the target instruction n times.
 * Using explicit functions (rather than a macro loop) keeps the
 * instruction body in the measured region and avoids function-call
 * overhead inside the hot loop while still allowing a function-pointer
 * dispatch table for the batch rotation.
 * ========================================================= */
#define STEP5_ITERS   1000
#define STEP5_WARMUP   100
#define STEP5_N_INSTR    6

static void s5_nop  (int n) {
    for (int i = 0; i < n; i++) asm volatile("add x0,x0,x0":::);
}
static void s5_add  (int n) {
    uint64_t x = 1, y = 2;
    for (int i = 0; i < n; i++) asm volatile("add %0,%0,%1":"+r"(x):"r"(y));
}
static void s5_mul  (int n) {
    uint64_t x = 1, y = 2;
    for (int i = 0; i < n; i++) asm volatile("mul %0,%0,%1":"+r"(x):"r"(y));
}
static void s5_div  (int n) {
    uint64_t x = 0xABCDEF, y = 7;
    for (int i = 0; i < n; i++) asm volatile("div %0,%0,%1":"+r"(x):"r"(y));
}
static void s5_load (int n) {
    volatile uint64_t m = 0x12345678; uint64_t x;
    for (int i = 0; i < n; i++)
        asm volatile("ld %0,0(%1)":"=r"(x):"r"((uint64_t)&m));
}
static void s5_store(int n) {
    volatile uint64_t m; uint64_t x = 0xABCDEF;
    for (int i = 0; i < n; i++)
        asm volatile("sd %0,0(%1)"::"r"(x),"r"((uint64_t)&m):"memory");
}

typedef void (*s5_fn_t)(int);
static const s5_fn_t  s5_fns [STEP5_N_INSTR] = {
    s5_nop, s5_add, s5_mul, s5_div, s5_load, s5_store
};
static const char    *s5_name[STEP5_N_INSTR] = {
    "NOP", "ADD", "MUL", "DIV", "LOAD", "STORE"
};

/* =========================================================
 * Main
 * ========================================================= */
int main(void)
{
    int cpu = 3;

    puts("╔══════════════════════════════════════════════════════════════════════╗");
    puts("║  PMU Verification — EIC7700X (ESWIN 770x core)                     ║");
    puts("╚══════════════════════════════════════════════════════════════════════╝");
    putchar('\n');

    pin_to_cpu(cpu);
    printf("  Pinned to CPU %d\n\n", cpu);

    int fd = open("/dev/pmu_monitor", O_RDWR);
    if (fd < 0) {
        perror("  open /dev/pmu_monitor");
        puts("  → run: sudo insmod pmu_module.ko");
        return 1;
    }
    puts("  /dev/pmu_monitor opened");

    /* ─────────────────────────────────────────────────────────────────────
     * Step 1 — Legacy PMU setup (cycle, instret, 3 perf-based HPM events)
     * ───────────────────────────────────────────────────────────────────── */
    puts("\n[ Step 1 ] Legacy PMU setup (IOCTL_PMU_SETUP)");
    sep();

    struct pmu_setup_info setup = { .cpu = cpu };
    if (ioctl(fd, IOCTL_PMU_SETUP, &setup) < 0) {
        perror("  ioctl IOCTL_PMU_SETUP");
        close(fd);
        return 1;
    }

    printf("  scounteren     : 0x%08lx\n", setup.scounteren_val);
    printf("  cycle  (0xc00) : %s\n", tick(setup.cycle_ok));
    printf("  instret(0xc02) : %s\n", tick(setup.instret_ok));
    printf("  branch_miss    : %s  [hpmcounter%d  0x%03x]\n",
           tick(setup.branch_miss_ok), setup.branch_miss_idx,
           0xc00 + setup.branch_miss_idx);
    printf("  cache_miss     : %s  [hpmcounter%d  0x%03x]\n",
           tick(setup.cache_miss_ok), setup.cache_miss_idx,
           0xc00 + setup.cache_miss_idx);
    printf("  cache_ref      : %s  [hpmcounter%d  0x%03x]\n",
           tick(setup.cache_ref_ok), setup.cache_ref_idx,
           0xc00 + setup.cache_ref_idx);
    printf("  counters live  : %s\n", tick(setup.counters_running));

    int step1_ok = setup.cycle_ok && setup.instret_ok && setup.counters_running;

    /* ─────────────────────────────────────────────────────────────────────
     * Step 2 — SBI raw event scan
     *
     * Iterates 16 classes × 32 single-bit mask positions = 512 raw event
     * encodings via SBI CFG_MATCH.  Reports which ones the firmware accepts.
     * ───────────────────────────────────────────────────────────────────── */
    puts("\n[ Step 2 ] SBI raw event scan (IOCTL_PMU_RAW_SCAN)");
    sep();

    struct pmu_raw_scan_result raw_scan;
    memset(&raw_scan, 0, sizeof(raw_scan));
    int scan_ok = (ioctl(fd, IOCTL_PMU_RAW_SCAN, &raw_scan) == 0);

    if (!scan_ok) {
        puts("  IOCTL_PMU_RAW_SCAN failed — module too old, update pmu_module.ko");
    } else if (raw_scan.count == 0) {
        puts("  No events found.  SBI raw event type may not be supported by firmware.");
        puts("  All probes returned SBI_ERR_NOT_SUPPORTED or SBI_ERR_INVALID_PARAM.");
    } else {
        printf("  Found %d valid event encodings:\n\n", raw_scan.count);
        printf("  %-4s  %-12s  %-5s  %-10s  %-6s\n",
               "Idx", "event_data", "ctr", "class/bit", "CSR");
        printf("  %-4s  %-12s  %-5s  %-10s  %-6s\n",
               "----", "----------", "-----", "----------", "------");
        for (int i = 0; i < raw_scan.count; i++) {
            uint64_t ed    = raw_scan.events[i].event_data;
            int      ctr   = raw_scan.events[i].ctr_idx;
            uint8_t  cls   = (uint8_t)(ed & 0xFF);
            int      bit   = -1;
            for (int b = 8; b < 64; b++) {
                if (ed >> b & 1) { bit = b - 8; break; }
            }
            printf("  [%2d]  0x%08llx    %3d  class=0x%02x bit=%-2d  0x%03x\n",
                   i, (unsigned long long)ed, ctr, cls, bit,
                   0xc00 + ctr);
        }
    }

    /* Open all discovered events so counters run during subsequent steps */
    struct pmu_open_raw_request open_req;
    memset(&open_req, 0, sizeof(open_req));
    open_req.cpu   = cpu;
    open_req.count = raw_scan.count;
    for (int i = 0; i < raw_scan.count; i++)
        open_req.event_data[i] = raw_scan.events[i].event_data;

    if (raw_scan.count > 0)
        ioctl(fd, IOCTL_PMU_OPEN_RAW, &open_req);

    int step2_ok = scan_ok;

    /* ─────────────────────────────────────────────────────────────────────
     * Step 3 — U-mode CSR accessibility
     *
     * For each discovered raw counter: attempt a csrr from U-mode.
     * A SIGILL means scounteren bit is not set for that counter.
     * ───────────────────────────────────────────────────────────────────── */
    puts("\n[ Step 3 ] U-mode CSR accessibility");
    sep();

    /* Fixed counters first */
    {
        uint64_t v;
        int ok;

        ok = 1;
        asm volatile("csrr %0,0xc00":"=r"(v));
        printf("  0xc00  cycle   : %s  (val=%lu)\n", ok ? "✓ readable" : "✗ SIGILL", v);

        ok = 1;
        asm volatile("csrr %0,0xc02":"=r"(v));
        printf("  0xc02  instret : %s  (val=%lu)\n", ok ? "✓ readable" : "✗ SIGILL", v);
    }

    /* Raw-discovered counters */
    int all_raw_readable = 1;
    for (int i = 0; i < raw_scan.count; i++) {
        int   ctr = raw_scan.events[i].ctr_idx;
        int   csr = 0xc00 + ctr;
        uint64_t val = 0;
        int   ok  = probe_csr_readable(ctr, &val);
        printf("  0x%03x  hpm%-2d   : %s  event_data=0x%08llx\n",
               csr, ctr,
               ok ? "✓ readable" : "✗ SIGILL — scounteren bit missing",
               (unsigned long long)raw_scan.events[i].event_data);
        if (!ok) all_raw_readable = 0;
    }

    if (raw_scan.count == 0)
        puts("  (no raw events to probe)");

    int step3_ok = all_raw_readable;

    /* ─────────────────────────────────────────────────────────────────────
     * Step 4 — Counter liveness (using OPEN_RAW counter assignments)
     *
     * The scan's ctr_idx is a probe-time artifact: the probe-release loop
     * reuses the same counter for every event, so all 29 events appear on
     * the same CSR.  The actual counter assignment comes from OPEN_RAW.
     *
     * Firmware limit: this board exposes only 3 HPM counters via SBI
     * (hpm4, hpm5, hpm6).  OPEN_RAW opens one event per counter, so only
     * the first 3 events get real measurements here.  The remaining 26
     * are marked "not opened" — test them by re-running with a different
     * starting event index, or close/open individual events between runs.
     * ───────────────────────────────────────────────────────────────────── */
    puts("\n[ Step 4 ] Counter liveness (OPEN_RAW assignments)");
    sep();

    /* Count how many events were actually opened */
    int n_opened = 0;
    for (int i = 0; i < raw_scan.count; i++)
        if (open_req.ok[i]) n_opened++;

    printf("  Firmware HPM counters available: %d  (hpm4–hpm6 on this board)\n", n_opened);
    printf("  Events opened: %d / %d  (%d need a separate run)\n\n",
           n_opened, raw_scan.count, raw_scan.count - n_opened);

    /* Fixed counters */
    {
        uint64_t cy_b, cy_a, ir_b, ir_a;
        asm volatile("csrr %0,0xc00":"=r"(cy_b));
        asm volatile("csrr %0,0xc02":"=r"(ir_b));
        workload_nop();
        asm volatile("csrr %0,0xc00":"=r"(cy_a));
        asm volatile("csrr %0,0xc02":"=r"(ir_a));
        printf("  %-10s  ctr=%-2s  NOP=%-10lu  BRANCH=%-10s  CACHE=%-10s  %s\n",
               "cycle", "-", cy_a - cy_b, "(n/a)", "(n/a)", cy_a > cy_b ? "✓ live" : "✗");
        printf("  %-10s  ctr=%-2s  NOP=%-10lu  BRANCH=%-10s  CACHE=%-10s  %s\n",
               "instret", "-", ir_a - ir_b, "(n/a)", "(n/a)", ir_a > ir_b ? "✓ live" : "✗");
    }

    printf("\n  %-12s  %-4s  %-12s  %-12s  %-12s  %s\n",
           "event_data", "ctr", "NOP_delta", "BRANCH_delta", "CACHE_delta", "Status");
    printf("  %-12s  %-4s  %-12s  %-12s  %-12s  %s\n",
           "----------", "---", "---------", "------------", "-----------", "------");

    int n_live   = 0;
    int n_frozen = 0;

    for (int i = 0; i < raw_scan.count; i++) {
        uint64_t ed = raw_scan.events[i].event_data;

        if (!open_req.ok[i]) {
            printf("  0x%08llx    ---  (not opened — insufficient counters)\n",
                   (unsigned long long)ed);
            continue;
        }

        int ctr = open_req.ctr_idx[i];  /* ← actual assigned counter, not probe artifact */

        uint64_t dummy;
        if (!probe_csr_readable(ctr, &dummy)) {
            printf("  0x%08llx    %3d  (SIGILL — scounteren bit not set)\n",
                   (unsigned long long)ed, ctr);
            continue;
        }

        uint64_t d_nop    = measure_delta(ctr, workload_nop);
        uint64_t d_branch = measure_delta(ctr, workload_branch);
        uint64_t d_cache  = measure_delta(ctr, workload_cache);
        int      live     = (d_nop > 0 || d_branch > 0 || d_cache > 0);

        printf("  0x%08llx    %3d  %-12lu  %-12lu  %-12lu  %s\n",
               (unsigned long long)ed, ctr, d_nop, d_branch, d_cache,
               live ? "✓ live" : "~ zero (commit-only or wrong workload)");

        if (live) n_live++;
        else      n_frozen++;
    }

    printf("\n  Opened and live: %d   Opened but zero: %d   Not opened: %d\n",
           n_live, n_frozen, raw_scan.count - n_opened);
    if (n_frozen > 0)
        puts("  ^ Zero-delta opened counters: event may require a different workload,\n"
             "    or this confirms commit-only (no speculative) counting on this CPU.");

    /* ─────────────────────────────────────────────────────────────────────
     * Step 4b — Targeted single-event tests
     *
     * Close all open events, then test each key event in isolation:
     * open it alone on counter 4, run the matching workload, report delta.
     * This avoids the counter-sharing problem in Step 4 and gives a clean
     * per-event verdict.
     *
     * Key events tested:
     *   0x2001 branch_direction_misprediction  → BRANCH workload
     *   0x4001 branch_target_misprediction     → BRANCH workload
     *   0x8001 pipeline_flush                  → BRANCH workload
     *   0x0100 exception_taken                 → CACHE  workload (TLB faults)
     *   0x0200 integer_load_retired            → NOP    workload
     * ───────────────────────────────────────────────────────────────────── */
    puts("\n[ Step 4b ] Targeted single-event tests");
    sep();

    struct {
        uint64_t    event_data;
        const char *name;
        void       (*workload)(void);
    } targets[] = {
        { 0x00002001, "branch_dir_mispredict", workload_branch },
        /* use an indirect-branch workload to provoke branch-target misses */
        { 0x00004001, "branch_tgt_mispredict", workload_indirect_branch },
        /* return/indirect sequences to exercise pipeline/return mispredicts */
        { 0x00008001, "pipeline_flush",        workload_return_mispredict },
        { 0x00000100, "exception_taken",        workload_cache  },
        { 0x00000200, "int_load_retired",       workload_nop    },
    };
    int n_targets = (int)(sizeof(targets) / sizeof(targets[0]));

    /* Close current open events to free all counters */
    ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);

    printf("  %-26s  %-12s  %-10s  %s\n",
           "Event", "event_data", "delta", "Status");
    printf("  %-26s  %-12s  %-10s  %s\n",
           "--------------------------", "----------", "----------", "------");

    for (int t = 0; t < n_targets; t++) {
        struct pmu_open_raw_request req1;
        memset(&req1, 0, sizeof(req1));
        req1.cpu          = cpu;
        req1.count        = 1;
        req1.event_data[0] = targets[t].event_data;

        if (ioctl(fd, IOCTL_PMU_OPEN_RAW, &req1) < 0 || !req1.ok[0]) {
            printf("  %-26s  0x%08llx    %-10s  OPEN_FAIL\n",
                   targets[t].name,
                   (unsigned long long)targets[t].event_data, "-");
            continue;
        }

        int ctr = req1.ctr_idx[0];
        uint64_t d = measure_delta(ctr, targets[t].workload);

        printf("  %-26s  0x%08llx    %-10lu  %s\n",
               targets[t].name,
               (unsigned long long)targets[t].event_data,
               d,
               d > 0 ? "✓ live" : "~ zero");

        ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 5 — Full PMU characterization via batch rotation
     *
     * With only 4 hardware HPM counters (hpm3–6) but 312 valid event
     * encodings, we rotate through events in batches of PMU_RAW_CTR_MAX.
     *
     * For each batch of ≤4 events:
     *   - IOCTL_PMU_OPEN_RAW with the batch subset
     *   - For each instruction type: warmup, snapshot, N iters, snapshot
     *   - Record HPM deltas for this batch
     *   - IOCTL_PMU_CLOSE_RAW
     *
     * Cycle and instret are free counters; measured once per instruction
     * type in the first batch (no HPM interference).
     *
     * Output:
     *   - Summary table: per-instruction cycle/instret + non-zero events
     *   - Full CSV written to pmu_step5.csv for offline analysis
     * ───────────────────────────────────────────────────────────────────── */
    puts("\n[ Step 5 ] Full PMU characterization (batch rotation over all events)");
    sep();

    /* Result storage — declare static to avoid 24 KB stack frame */
    static uint64_t s5_cycles [STEP5_N_INSTR];
    static uint64_t s5_instret[STEP5_N_INSTR];
    static uint64_t s5_deltas [STEP5_N_INSTR][PMU_RAW_MAX_EVENTS];
    memset(s5_cycles,  0, sizeof(s5_cycles));
    memset(s5_instret, 0, sizeof(s5_instret));
    memset(s5_deltas,  0, sizeof(s5_deltas));

    int n_batches   = (raw_scan.count + PMU_RAW_CTR_MAX - 1) / PMU_RAW_CTR_MAX;
    int first_batch = 1;

    printf("  Rotating %d events across %d instruction types in %d batches of ≤%d...\n\n",
           raw_scan.count, STEP5_N_INSTR, n_batches, PMU_RAW_CTR_MAX);

    for (int bs = 0; bs < raw_scan.count; bs += PMU_RAW_CTR_MAX) {
        int bsz = raw_scan.count - bs;
        if (bsz > PMU_RAW_CTR_MAX) bsz = PMU_RAW_CTR_MAX;

        /* Open this batch */
        struct pmu_open_raw_request breq;
        memset(&breq, 0, sizeof(breq));
        breq.cpu   = cpu;
        breq.count = bsz;
        for (int j = 0; j < bsz; j++)
            breq.event_data[j] = raw_scan.events[bs + j].event_data;

        if (ioctl(fd, IOCTL_PMU_OPEN_RAW, &breq) < 0)
            goto next_batch;

        /* Measure each instruction type against this batch */
        for (int it = 0; it < STEP5_N_INSTR; it++) {
            uint64_t hb[PMU_RAW_CTR_MAX] = {0};
            uint64_t ha[PMU_RAW_CTR_MAX] = {0};
            uint64_t cy_b, cy_a, ir_b, ir_a;

            s5_fns[it](STEP5_WARMUP);   /* cache/branch-predictor warmup */

            asm volatile("csrr %0,0xc00":"=r"(cy_b));
            asm volatile("csrr %0,0xc02":"=r"(ir_b));
            for (int j = 0; j < bsz; j++)
                if (breq.ok[j])
                    hb[j] = read_hpmcounter(breq.ctr_idx[j]);

            s5_fns[it](STEP5_ITERS);    /* measured region */

            asm volatile("csrr %0,0xc00":"=r"(cy_a));
            asm volatile("csrr %0,0xc02":"=r"(ir_a));
            for (int j = 0; j < bsz; j++)
                if (breq.ok[j])
                    ha[j] = read_hpmcounter(breq.ctr_idx[j]);

            /* Cycle/instret taken from first batch only (lowest SBI overhead) */
            if (first_batch) {
                s5_cycles [it] = cy_a - cy_b;
                s5_instret[it] = ir_a - ir_b;
            }
            for (int j = 0; j < bsz; j++)
                if (breq.ok[j])
                    s5_deltas[it][bs + j] = ha[j] - hb[j];
        }

        ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);
        first_batch = 0;
        continue;
next_batch:
        ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);
    }

    /* ── Print summary ── */
    printf("  %-6s  %8s  %8s  Notable events (event_data → delta)\n",
           "Instr", "cycles", "instret");
    printf("  %-6s  %8s  %8s  %s\n",
           "------", "--------", "--------",
           "------------------------------------------------------------");

    for (int it = 0; it < STEP5_N_INSTR; it++) {
        printf("  %-6s  %8lu  %8lu",
               s5_name[it], s5_cycles[it], s5_instret[it]);
        int nz = 0;
        for (int ev = 0; ev < raw_scan.count; ev++) {
            if (s5_deltas[it][ev] > 0) {
                printf("  [0x%llx→%lu]",
                       (unsigned long long)raw_scan.events[ev].event_data,
                       s5_deltas[it][ev]);
                nz++;
            }
        }
        if (nz == 0) printf("  (no non-zero events)");
        putchar('\n');
    }

    /* ── Write full CSV ── */
    FILE *csv = fopen("pmu_step5.csv", "w");
    if (csv) {
        /* Header: instr, cycles, instret, then one column per event */
        fprintf(csv, "instr,cycles,instret");
        for (int ev = 0; ev < raw_scan.count; ev++)
            fprintf(csv, ",0x%llx",
                    (unsigned long long)raw_scan.events[ev].event_data);
        fprintf(csv, "\n");

        /* One row per instruction type */
        for (int it = 0; it < STEP5_N_INSTR; it++) {
            fprintf(csv, "%s,%lu,%lu",
                    s5_name[it], s5_cycles[it], s5_instret[it]);
            for (int ev = 0; ev < raw_scan.count; ev++)
                fprintf(csv, ",%lu", s5_deltas[it][ev]);
            fprintf(csv, "\n");
        }
        fclose(csv);
        puts("\n  Full data written to pmu_step5.csv");
    } else {
        puts("\n  (could not write pmu_step5.csv)");
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 6 — Teardown
     * ───────────────────────────────────────────────────────────────────── */
    putchar('\n');
    ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);
    if (ioctl(fd, IOCTL_PMU_TEARDOWN, NULL) < 0)
        perror("  IOCTL_PMU_TEARDOWN (non-fatal)");
    else
        puts("[ Step 6 ] PMU teardown complete");
    close(fd);

    /* ─────────────────────────────────────────────────────────────────────
     * Summary
     * ───────────────────────────────────────────────────────────────────── */
    puts("\n╔══════════════════════════════════════════════════════════════════════╗");
    puts("║  Summary                                                            ║");
    puts("╚══════════════════════════════════════════════════════════════════════╝");
    printf("  Step 1 — Legacy setup    : %s\n", tick(step1_ok));
    printf("  Step 2 — Raw scan        : %s  (%d events found)\n",
           tick(step2_ok && raw_scan.count > 0), raw_scan.count);
    printf("  Step 3 — CSR readable    : %s\n", tick(step3_ok));
    printf("  Step 4 — Counter liveness: %d live  %d zero-delta\n",
           n_live, n_frozen);
    if (n_frozen > 0)
        puts("           ^ zero-delta counters likely commit-only (no speculative counting)");

    int pass = step1_ok && step2_ok && step3_ok;
    printf("\n  Overall: %s\n\n", pass ? "PASS ✓" : "FAIL ✗  — see steps above");
    return pass ? 0 : 1;
}