// SPDX-License-Identifier: GPL-2.0 OR MIT
/*
 * pmu_module.c — RISC-V PMU setup driver for EIC7700X (ESWIN 770x core)
 *
 * Sole responsibility: configure hardware performance counters so that
 * user-space can read them directly via CSR instructions with zero kernel
 * involvement inside the measurement loop.
 *
 * All measurement logic lives entirely in user-space (pmu_profiler / scanner).
 * This module only handles the one-time setup that requires kernel/firmware
 * access, and the corresponding teardown.
 *
 * Counters exposed to U-mode after IOCTL_PMU_SETUP:
 *   cycle    (0xc00) — clock cycles
 *   instret  (0xc02) — instructions retired
 *
 * Additional HPM counters are configured dynamically via the raw scan IOCTLs:
 *   IOCTL_PMU_RAW_SCAN   — probe all 512 class/mask combinations via SBI CFG_MATCH
 *                          and return a table of every valid event encoding with
 *                          its assigned counter index.
 *   IOCTL_PMU_OPEN_RAW   — open a caller-specified list of raw events for measurement;
 *                          leaves counters running so user-space can read them via csrr.
 *   IOCTL_PMU_CLOSE_RAW  — stop and release all raw-opened counters.
 *
 * Raw event encoding format (EIC7700X / SiFive class-mask scheme):
 *   mhpmevent[7:0]   = event class  (0x00 = instructions, 0x01 = microarch, 0x02 = cache…)
 *   mhpmevent[63:8]  = event mask   (one bit per event within the class)
 *
 * The scan iterates 16 classes × 32 single-bit mask positions = 512 values.
 * This covers every documented and undocumented event that follows the SiFive
 * encoding convention, as described in the EIC7700X TRM §3.4.1 and §3.5 (pL2 PMU).
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/miscdevice.h>
#include <linux/string.h>
#include <linux/smp.h>
#include <linux/perf_event.h>
#include <linux/ktime.h>
#include <asm/sbi.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("ESWIN PMU");
MODULE_DESCRIPTION("PMU counter setup + raw event scan for U-mode access on EIC7700X");

#define DEVICE_NAME "pmu_monitor"

/* =========================================================
 * SBI PMU Extension — EID 0x504D55 ("PMU"), SBI spec v2.0 §11
 * ========================================================= */
#define SBI_EXT_PMU                   0x504D55

/* SBI PMU Function IDs */
#define SBI_EXT_PMU_FID_NUM_COUNTERS  0  /* how many counters exist         */
#define SBI_EXT_PMU_FID_GET_CTR_INFO  1  /* info about one counter          */
#define SBI_EXT_PMU_FID_CFG_MATCH     2  /* allocate counter for an event   */
#define SBI_EXT_PMU_FID_START         3  /* clear mcountinhibit → start     */
#define SBI_EXT_PMU_FID_STOP          4  /* set   mcountinhibit → stop      */

/* CFG_MATCH config_flags (SBI spec v3.0 §12.1.8, Table 32):
 *   bit 0 = SBI_PMU_CFG_FLAG_SKIP_MATCH   bypass DTB event-to-counter matching
 *   bit 1 = SBI_PMU_CFG_FLAG_CLEAR_VALUE  zero counter value on configure
 *   bit 2 = SBI_PMU_CFG_FLAG_AUTO_START   start counter after configuring
 *   bit 3 = SBI_PMU_CFG_FLAG_SET_VUINH    inhibit counting in VU-mode
 *   bit 4 = SBI_PMU_CFG_FLAG_SET_VSINH    inhibit counting in VS-mode
 *   bit 5 = SBI_PMU_CFG_FLAG_SET_UINH     inhibit counting in U-mode
 *   bit 6 = SBI_PMU_CFG_FLAG_SET_SINH     inhibit counting in S-mode
 *   bit 7 = SBI_PMU_CFG_FLAG_SET_MINH     inhibit counting in M-mode
 *   bits 8+ = RESERVED — must be zero or SBI_ERR_INVALID_PARAM is returned
 *
 * AUTO_START | CLEAR_VALUE combined = configure + zero + start in one SBI call,
 * eliminating the need for a separate sbi_pmu_counter_start() call.
 */
#define PMU_CFG_FLAG_SKIP_MATCH       (1UL << 0)
#define PMU_CFG_FLAG_CLEAR_VALUE      (1UL << 1)
#define PMU_CFG_FLAG_AUTO_START       (1UL << 2)
#define PMU_CFG_FLAG_OPEN             (PMU_CFG_FLAG_AUTO_START | PMU_CFG_FLAG_CLEAR_VALUE)

/* START flags */
#define PMU_START_SET_INIT_VALUE      (1UL << 0)  /* zero counter on start   */

/* STOP flags */
#define PMU_STOP_FLAG_RESET           (1UL << 0)  /* stop AND release counter */

/*
 * SBI PMU raw event type.
 *
 * From the SBI spec v2.0 §11 event type table:
 *   0x0 = SBI_PMU_EVENT_TYPE_HW        (general hardware — requires DTB mapping)
 *   0x1 = SBI_PMU_EVENT_TYPE_HW_CACHE  (cache hardware — requires DTB mapping)
 *   0x2 = SBI_PMU_EVENT_TYPE_HW_RAW   ← hardware raw: event_data written verbatim
 *                                         to mhpmeventX, no DTB lookup
 *   0xF = SBI_PMU_EVENT_TYPE_FW        (firmware software counters — NOT mhpmeventX)
 *
 * Linux uses RISCV_PMU_RAW_EVENT_IDX = 0x20000 = (0x2 << 16) for PERF_TYPE_RAW
 * events.  We must match this exactly — 0xF is the firmware type and OpenSBI
 * will reject any event_data that does not match a known firmware event code.
 */
#define SBI_PMU_EVENT_TYPE_HW_RAW     0x2UL
#define SBI_PMU_RAW_EVENT_IDX         (SBI_PMU_EVENT_TYPE_HW_RAW << 16)  /* = 0x20000 */

/*
 * HPM counter bitmask covering hpmcounter3-31 (29 counters).
 * Used as counter_idx_mask with counter_idx_base = 0, so bit N = counter N.
 */
#define PMU_ALL_HPM_MASK              0xFFFFFFF8UL   /* bits 3-31 */
/*
 * PMU_RAW_MAX_EVENTS: maximum number of distinct valid event encodings
 * the scan can discover.  This is the scan table capacity — it is NOT
 * the number of hardware HPM counters (which is 4: hpm3–hpm6 on this board).
 * The scan probes 16 classes × 32 bit positions = 512 possible encodings;
 * each valid one is recorded here.  Setting this to 29 previously caused
 * the scan to stop after the first 29 events (classes 0x00–0x01 only),
 * silently missing all class 0x02+ events (cache, TLB, etc.) that are
 * also valid according to the firmware DTB.
 */
#define PMU_RAW_MAX_EVENTS   512   /* scan table capacity */
#define PMU_RAW_CTR_MAX        4   /* hardware HPM counters: hpm3–hpm6 */

/* =========================================================
 * IOCTL interface — must match definitions in user-space
 * ========================================================= */

/*
 * pmu_setup_info — IOCTL_PMU_SETUP (unchanged from original)
 * Sets up cycle/instret via SBI START and grants U-mode CSR access
 * via scounteren.  Also opens 3 perf-based HPM events for backward
 * compatibility with the existing scanner.
 */
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

/*
 * pmu_raw_event — one entry in the scan result table.
 */
struct pmu_raw_event {
    uint64_t event_data;  /* raw mhpmevent value that was accepted by firmware */
    int      ctr_idx;     /* hardware counter index (3-31) assigned by firmware */
};

/*
 * pmu_raw_scan_result — IOCTL_PMU_RAW_SCAN output.
 * Kernel fills this after probing all 512 class/mask combinations.
 */
struct pmu_raw_scan_result {
    int                  count;
    struct pmu_raw_event events[PMU_RAW_MAX_EVENTS];
};

/*
 * pmu_open_raw_request — IOCTL_PMU_OPEN_RAW.
 * User fills cpu, count, and event_data[].
 * Kernel fills ctr_idx[] and ok[].
 */
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
 * Global state
 * ========================================================= */

/* Perf event handles — no longer used after pmu_do_setup was stripped of HPM
 * perf events, but kept to avoid breaking the teardown path. */
static struct perf_event *g_perf_ev[3];

/* Bitmask of HPM counters opened via IOCTL_PMU_OPEN_RAW. */
static uint32_t g_raw_ctr_mask;

/*
 * Valid HPM counter mask discovered by IOCTL_PMU_RAW_SCAN.
 *
 * Excludes counter 3 (hpmcounter3 appears non-functional on this hardware)
 * and any counter indices the firmware does not expose.  Set once by
 * pmu_do_raw_scan() and reused by pmu_do_open_raw() so both operations
 * target the same working counters.
 *
 * Initialised to bits 4-31 as a safe default; the scan narrows it to only
 * the indices the firmware actually accepts.
 */
static unsigned long g_hpm_scan_mask = 0xFFFFFFF0UL;  /* bits 4-31 (skip ctr 3) */

/* =========================================================
 * scounteren — grant U-mode read access to all HPM CSRs
 * ========================================================= */
static void _set_scounteren_cpu(void *val_ptr)
{
    unsigned long val = *(unsigned long *)val_ptr;
    asm volatile ("csrw 0x106, %0" :: "r"(val));
}

static void set_scounteren_all_cpus(unsigned long val)
{
    on_each_cpu(_set_scounteren_cpu, &val, 1);
    pr_info("PMU: scounteren = 0x%08lx set on all %d CPUs\n",
            val, num_online_cpus());
}

/* =========================================================
 * SBI counter start / stop helpers
 *
 * Mask semantics: always use base=0 so that bit N in mask
 * directly addresses counter N.
 * ========================================================= */
static int pmu_start(unsigned long cidx)
{
    struct sbiret ret = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_START,
                                  0, 1UL << cidx,
                                  PMU_START_SET_INIT_VALUE,
                                  0, 0, 0);
    if (ret.error == SBI_ERR_ALREADY_STARTED) {
        pr_info("PMU: counter %lu already running\n", cidx);
        return 0;
    }
    if (ret.error) {
        pr_warn("PMU: START(cidx=%lu) failed: %ld\n", cidx, ret.error);
        return -1;
    }
    return 0;
}

static int pmu_stop(unsigned long cmask)
{
    struct sbiret ret = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_STOP,
                                  0, cmask, 0, 0, 0, 0);
    if (ret.error && ret.error != SBI_ERR_ALREADY_STOPPED) {
        pr_warn("PMU: STOP(mask=0x%lx) failed: %ld\n", cmask, ret.error);
        return -1;
    }
    return 0;
}

/* =========================================================
 * Kernel perf event helper (legacy path for IOCTL_PMU_SETUP)
 * ========================================================= */
#define EIC7700X_BRANCH_DIR_MISPREDICT  0x2001UL

static struct perf_event *pmu_open_perf_event(uint64_t type, uint64_t config, int cpu)
{
    struct perf_event_attr attr = {
        .type           = type,
        .size           = sizeof(attr),
        .config         = config,
        .disabled       = 0,
        .exclude_kernel = 0,
        .exclude_user   = 0,
        .exclude_hv     = 0,
        .pinned         = 1,
    };
    return perf_event_create_kernel_counter(&attr, cpu, NULL, NULL, NULL);
}

/* =========================================================
 * IOCTL_PMU_SETUP — minimal fixed-counter setup
 *
 * Configures cycle (hpmcounter0), instret (hpmcounter2) and
 * scounteren only.  HPM counter allocation is now handled
 * entirely by IOCTL_PMU_RAW_SCAN + IOCTL_PMU_OPEN_RAW.
 *
 * The branch_miss/cache_miss/cache_ref fields are left zeroed
 * (ok=0, idx=0).  Callers that relied on them should use the
 * raw scan path instead.
 * ========================================================= */
static void pmu_do_setup(struct pmu_setup_info *info)
{
    uint64_t c1, c2;
    int cpu, i;

    cpu = (info->cpu >= 0) ? info->cpu : 0;
    memset(info, 0, sizeof(*info));
    info->cpu = cpu;

    /* Grant U-mode read access to all counter CSRs on every hart */
    info->scounteren_val = 0xFFFFFFFF;
    set_scounteren_all_cpus(info->scounteren_val);

    /* Start fixed counters via SBI (clears mcountinhibit bits) */
    info->cycle_ok   = (pmu_start(0) == 0);
    info->instret_ok = (pmu_start(2) == 0);
    pr_info("PMU: cycle=%s instret=%s\n",
            info->cycle_ok   ? "ok" : "FAIL",
            info->instret_ok ? "ok" : "FAIL");

    /* Snapshot initial counter values */
    asm volatile ("csrr %0, 0xc00" : "=r"(info->cycle_initial));
    asm volatile ("csrr %0, 0xc02" : "=r"(info->instret_initial));

    /* Sanity check: verify cycle counter is incrementing */
    asm volatile ("csrr %0, 0xc00" : "=r"(c1));
    for (i = 0; i < 200; i++) asm volatile ("nop");
    asm volatile ("csrr %0, 0xc00" : "=r"(c2));

    info->counters_running = (c2 > c1);
    if (info->counters_running)
        pr_info("PMU: SANITY PASS (cycle delta=%llu)\n", c2 - c1);
    else
        pr_warn("PMU: SANITY FAIL — cycle counter frozen\n");
}

/* =========================================================
 * IOCTL_PMU_RAW_SCAN — discover all valid raw event encodings
 *
 * Mirrors what Linux does in pmu_sbi_get_ctrinfo() + pmu_sbi_check_event():
 *
 *   Step 1: Query NUM_COUNTERS to find how many logical counters exist.
 *   Step 2: Query GET_CTR_INFO for each counter to identify hardware HPM
 *           counters (type=0, csr 3-31) and build the valid mask — exactly
 *           as Linux builds rvpmu->cmask.
 *   Step 3: Iterate 512 class/mask event_data values.  For each, call
 *           CFG_MATCH with the firmware-validated mask and event_type=0x2
 *           (SBI_PMU_EVENT_TYPE_HW_RAW), which writes event_data verbatim
 *           to mhpmeventX without requiring a DTB event-to-mhpmevent entry.
 *
 * Why this was failing before:
 *   We used PMU_ALL_HPM_MASK = 0xFFFFFFF8 (bits 3-31), assuming 29 counters.
 *   If the firmware only exposes fewer logical counters, any bit beyond
 *   that range causes SBI_ERR_INVALID_PARAM for the entire call — even
 *   though the event_data and event_type are correct.
 *   Building the mask from GET_CTR_INFO matches rvpmu->cmask exactly.
 * ========================================================= */
/*
 * g_event_skip_list — known-noisy/unusable raw mhpmevent encodings.
 *
 * Checked once, inside pmu_do_raw_scan's Step 3 loop below.  Any
 * event_data value in this list is never added to result->events[],
 * so it never reaches IOCTL_PMU_RAW_SCAN's caller and is never opened
 * by IOCTL_PMU_OPEN_RAW.  Downstream code (pmu_scanner.c,
 * spec_helpers.c) requires no skip-list awareness at all — they simply
 * never see these events as having been discovered, by design, so no
 * per-iteration check is needed anywhere in the measurement hot path.
 *
 * Populate based on empirical findings: events that consistently fail
 * to settle (majority-vote noise) across many independent runs and
 * instructions, with no plausible per-instruction explanation, are
 * candidates for this list.  Document why each entry was added —
 * "noisy in practice" is a real but soft judgement call, not a hard
 * firmware fact, so a future re-evaluation (e.g. after a firmware
 * update, or after the settle-window parameters change) should be able
 * to revisit each entry on its own terms.
 */
static const uint64_t g_event_skip_list[] = {
    // 0x0027,
    // 0x0028,
};
#define N_EVENT_SKIP (int)(sizeof(g_event_skip_list) / sizeof(g_event_skip_list[0]))

static bool pmu_event_is_skipped(uint64_t event_data)
{
    for (int i = 0; i < N_EVENT_SKIP; i++)
        if (g_event_skip_list[i] == event_data)
            return true;
    return false;
}

/*
 * Probe a single candidate event_data against scan_mask via CFG_MATCH.
 * On success, records it in result and releases the counter back to the
 * firmware pool.  Shared by all event-space sweeps in pmu_do_raw_scan()
 * (P550-style class/bit, T-Head flat IDs, and the dense pilot sweep) since
 * the probe/record/release sequence is identical — only the candidate
 * event_data values, and volume, differ.
 *
 * Returns 1 if the event opened (regardless of whether it was actually
 * stored — result->count may already be at PMU_RAW_MAX_EVENTS), 0 otherwise,
 * so callers doing a large sweep can report a true hit count even once
 * storage is full rather than silently under-reporting.
 *
 * verbose controls the skip-list pr_info: kept on for the existing small
 * (512- and 42-candidate) sweeps for their existing diagnostic trace, off
 * for the dense pilot sweep where per-candidate logging at tens of
 * thousands of iterations would both flood dmesg and dominate the timing
 * measurement it's trying to produce. */
/*
 * Returns true if event_data is already present in result->events[].
 * The three raw-event sweeps (P550 class/bit, T-Head flat IDs, dense
 * 16-bit pilot) are intentionally over-inclusive and allowed to overlap
 * in *candidate* space — e.g. T-Head's 0x01-0x2a IDs are a strict subset
 * of the dense pilot's 0x0000-0xFFFF range. This guard prevents an
 * encoding that already opened in an earlier sweep from being recorded
 * a second time in result->events[].
 */
static bool pmu_result_has_event(const struct pmu_raw_scan_result *result,
                                  uint64_t event_data)
{
    for (int i = 0; i < result->count; i++) {
        if (result->events[i].event_data == event_data)
            return true;
    }
    return false;
}

static int pmu_probe_raw_candidate(unsigned long scan_mask, uint64_t event_data,
                                   struct pmu_raw_scan_result *result, bool verbose)
{
    struct sbiret ret;

    if (pmu_event_is_skipped(event_data)) {
        if (verbose)
            pr_info("PMU:   event_data=0x%04llx skipped (skip list)\n", event_data);
        return 0;
    }

    ret = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_CFG_MATCH,
                    0,                      /* counter_idx_base       */
                    scan_mask,              /* firmware-validated mask */
                    0,                      /* config_flags           */
                    SBI_PMU_RAW_EVENT_IDX,  /* = 0x20000 (HW raw)     */
                    event_data,             /* verbatim → mhpmeventX  */
                    0);

    if (ret.error)
        return 0;   /* this event isn't valid on this board — not a global signal */

    if (result->count < PMU_RAW_MAX_EVENTS &&
        !pmu_result_has_event(result, event_data)) {
        result->events[result->count].event_data = event_data;
        result->events[result->count].ctr_idx    = (int)ret.value;
        result->count++;
    }

    /*
     * Release counter back to firmware pool for subsequent probes.
     * Matches pmu_sbi_check_event() in the Linux kernel.
     */
    sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_STOP,
              ret.value, 0x1, PMU_STOP_FLAG_RESET, 0, 0, 0);
    return 1;
}

static void pmu_do_raw_scan(struct pmu_raw_scan_result *result)
{
    unsigned long scan_mask = 0;
    uint32_t class, bit;
    int sbi_supported = -1;
    int n_ctrs;

    memset(result, 0, sizeof(*result));

    /* ── Step 1: query firmware for total counter count ── */
    {
        struct sbiret ret = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_NUM_COUNTERS,
                                      0, 0, 0, 0, 0, 0);
        if (ret.error) {
            pr_warn("PMU: raw scan: NUM_COUNTERS failed: %ld\n", ret.error);
            return;
        }
        n_ctrs = (int)ret.value;
        pr_info("PMU: raw scan: firmware reports %d logical counters\n", n_ctrs);
    }

    /* ── Step 2: build mask of valid hardware HPM counters ──
     *
     * Counter info return value layout (union sbi_pmu_ctr_info in kernel):
     *   bit  63     = type: 0=hardware, 1=firmware
     *   bits [17:12] = width-1
     *   bits [11:0]  = CSR offset from 0xC00
     *                  (0=cycle, 1=time, 2=instret, 3-31=hpmcounterN)
     *
     * Include only hardware counters (type=0) with CSR offset 3-31.
     * Log every counter regardless so the dmesg is fully diagnostic.    */
    for (int i = 0; i < n_ctrs && i < 64; i++) {
        struct sbiret ret = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_GET_CTR_INFO,
                                      i, 0, 0, 0, 0, 0);
        if (ret.error) {
            pr_info("PMU:   ctr[%2d]: GET_CTR_INFO err=%ld\n", i, ret.error);
            continue;
        }

        unsigned long ctr_type = (ret.value >> 63) & 0x1UL;
        unsigned long ctr_csr  =  ret.value        & 0xFFFUL;
        unsigned long ctr_wid  = (ret.value >> 12)  & 0x3FUL;

        /*
         * ctr_csr is the absolute 12-bit CSR address (e.g., hpmcounter3 = 0xC03).
         * Hardware HPM counters: type=0, CSR in [0xC03, 0xC1F].
         * The logical counter index (used in counter_idx_mask) is (csr - 0xC00).
         * Previous code checked csr >= 3, which always failed since csr >= 0xC03.
         */
        int is_hpm = (ctr_type == 0 && ctr_csr >= 0xC03 && ctr_csr <= 0xC1F);

        pr_info("PMU:   ctr[%2d]: type=%lu csr=0x%03lx width=%lu val=0x%016lx %s\n",
                i, ctr_type, ctr_csr, ctr_wid, (unsigned long)ret.value,
                is_hpm ? "→ HPM" : "");

        if (is_hpm)
            scan_mask |= BIT(ctr_csr - 0xC00);  /* logical idx = CSR offset */
    }

    pr_info("PMU: raw scan: GET_CTR_INFO mask = 0x%08lx\n", scan_mask);

    /*
     * Fallback: if GET_CTR_INFO gave us nothing, probe via CFG_MATCH.
     * Start from counter 3 — GET_CTR_INFO confirms hpmcounter3 is hardware
     * type and valid.  The earlier exclusion of counter 3 was a mistake based
     * on a misread of the liveness test results.
     */
    if (!scan_mask) {
        pr_info("PMU: raw scan: GET_CTR_INFO gave empty mask — "
                "probing counter availability via CFG_MATCH\n");

        for (int i = 3; i <= 31; i++) {
            struct sbiret r = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_CFG_MATCH,
                                        0, BIT(i), 0,
                                        SBI_PMU_RAW_EVENT_IDX, 0x2001ULL, 0);
            if (r.error == 0) {
                pr_info("PMU:   ctr[%2d]: exists and free\n", i);
                scan_mask |= BIT(i);
                sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_STOP,
                          r.value, 0x1, PMU_STOP_FLAG_RESET, 0, 0, 0);
            } else if (r.error == SBI_ERR_ALREADY_STARTED) {
                pr_info("PMU:   ctr[%2d]: exists but busy\n", i);
                scan_mask |= BIT(i);
            } else {
                pr_info("PMU:   ctr[%2d]: err=%ld\n", i, r.error);
            }
        }

        pr_info("PMU: raw scan: probe-based mask = 0x%08lx\n", scan_mask);
    }

    /* Store the validated mask (counters 4+) for use by pmu_do_open_raw */
    g_hpm_scan_mask = scan_mask;

    if (!scan_mask) {
        pr_warn("PMU: raw scan: no valid HPM hardware counters found\n");
        return;
    }

    /* ── Step 3: probe every known event space and keep what the board accepts ──
     *
     * Two independent event-encoding conventions are tried, unconditionally and
     * without any vendor detection: the SiFive/P550-style class/bit field, and
     * T-Head's flat linear ID range (0x01-0x2a, documented for C906/C910/C920 —
     * see Linux kernel commit adding tools/perf/pmu-events/arch/riscv/thead/
     * c900-legacy/{cache,instruction}.json).  There is no need to pick one: an
     * event_data value invalid for the board simply fails CFG_MATCH here and is
     * skipped, which is the scanner's entire operating principle.  The two
     * ranges cannot collide (T-Head IDs top out at 0x2a; the P550 scheme's
     * minimum value is 0x100), so no dedup is needed. */
    for (class = 0x00; class <= 0x0F; class++) {
        for (bit = 0; bit < 32; bit++) {
            uint64_t event_data = ((uint64_t)1 << (bit + 8)) | class;
            pmu_probe_raw_candidate(scan_mask, event_data, result, true);
        }
    }
    for (uint64_t thead_id = 0x01; thead_id <= 0x2a; thead_id++)
        pmu_probe_raw_candidate(scan_mask, thead_id, result, true);

    /*
     * ── Pilot: dense sweep of the low 16 bits of event_data ──
     *
     * The two hand-curated spaces above (P550 class/bit, T-Head flat IDs)
     * are a tiny corner of the 64-bit event_data space — 554 candidates.
     * Rather than guess how much further is worth exploring, this dense
     * pass tries every value 0x0000-0xFFFF unconditionally and times it, so
     * the real probes/sec on this firmware is measured rather than assumed.
     * That throughput number is what decides whether extending to 20/24
     * bits afterwards costs minutes or hours.
     *
     * verbose=false: at 65536 candidates, per-candidate skip-list pr_info
     * would both flood dmesg and dominate the very timing this is meant to
     * produce.  hits_total counts real opens independent of result->count,
     * which stops recording at PMU_RAW_MAX_EVENTS — so the reported number
     * stays honest even if this pass finds more than the array can hold.
     */
    {
        uint64_t hits_total = 0;
        ktime_t t_start = ktime_get();

        for (uint64_t candidate = 0x0000; candidate <= 0xFFFF; candidate++)
            hits_total += pmu_probe_raw_candidate(scan_mask, candidate, result, false);

        s64 elapsed_ms = ktime_to_ms(ktime_sub(ktime_get(), t_start));

        pr_info("PMU: raw scan: dense 16-bit pilot: %llu/65536 opened in %lld ms "
                "(%llu probes/sec), %d/%d stored (capacity=%d)\n",
                hits_total, elapsed_ms,
                elapsed_ms > 0 ? (hits_total * 1000ULL) / (uint64_t)elapsed_ms : 0,
                result->count, PMU_RAW_MAX_EVENTS, PMU_RAW_MAX_EVENTS);
    }

    /*
     * sbi_supported used to latch onto whichever event_data happened to be
     * probed FIRST, which is not a reliable signal — an individual event
     * failing CFG_MATCH (wrong ID for this board's vendor/microarch) looks
     * identical to raw events being unsupported at all.  Empirically confirmed:
     * event_data=0x1 opens cleanly on this board even though earlier P550-style
     * IDs in the sweep returned NOT_SUPPORTED.  The only reliable signal for
     * "raw events don't work on this firmware at all" is that NOTHING in either
     * space opened. */
    sbi_supported = (result->count > 0) ? 1 : 0;

    if (sbi_supported == 0)
        pr_warn("PMU: raw scan: SBI raw event type not supported by firmware\n");
    else
        pr_info("PMU: raw scan: found %d valid events (sbi_supported=%d)\n",
                result->count, sbi_supported);

    for (int i = 0; i < result->count; i++)
        pr_info("PMU:   [%2d] event_data=0x%04llx → hpmcounter%d\n",
                i, result->events[i].event_data, result->events[i].ctr_idx);
}

/* =========================================================
 * IOCTL_PMU_CLOSE_RAW — stop and release all raw-opened counters
 * ========================================================= */
static void pmu_do_close_raw(void)
{
    if (!g_raw_ctr_mask)
        return;

    /*
     * Stop all counters in one SBI call.  Use RESET flag to release
     * them back to the firmware pool.
     */
    sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_STOP,
              0, g_raw_ctr_mask, PMU_STOP_FLAG_RESET, 0, 0, 0);

    pr_info("PMU: CLOSE_RAW released ctr_mask=0x%08x\n", g_raw_ctr_mask);
    g_raw_ctr_mask = 0;
}

/* =========================================================
 * IOCTL_PMU_OPEN_RAW — open a caller-specified set of raw events
 *
 * Two-phase allocation strategy:
 *
 * Phase 1 — DTB-validated (no SKIP_MATCH):
 *   Try the first N events using the scan-discovered mask (g_hpm_scan_mask).
 *   These are guaranteed to work on OpenSBI regardless of firmware version.
 *
 * Phase 2 — SKIP_MATCH extension:
 *   For events that didn't get a counter in Phase 1, try SBI_PMU_CFG_FLAG_SKIP_MATCH
 *   on each individual counter index 4–31 (one BIT at a time).  SKIP_MATCH
 *   instructs OpenSBI to bypass the DTB event-to-counter bitmask check
 *   and assign whatever hardware counter is free from the supplied mask.
 *   pmu_ctr_is_hw() in OpenSBI checks the physical hardware counter count,
 *   not the DTB count, so this can reach counters beyond the firmware's
 *   raw-event-to-mhpmcounters bitmask.
 *
 *   Using BIT(i) rather than a wide mask avoids SBI_ERR_INVALID_PARAM from
 *   OpenSBI implementations that validate the entire mask upfront — if a
 *   single-bit mask points to a non-existent counter the call fails cleanly
 *   and we move on.
 * ========================================================= */
static void pmu_do_open_raw(struct pmu_open_raw_request *req)
{
    int i, n;
    int n_phase1 = 0, n_phase2 = 0;

    pmu_do_close_raw();

    n = (req->count < PMU_RAW_MAX_EVENTS) ? req->count : PMU_RAW_MAX_EVENTS;

    for (i = 0; i < n; i++) {
        req->ok[i]      = 0;
        req->ctr_idx[i] = -1;
    }

    /* Positional pin: bind request slot i to hpmcounter(3+i) via a one-hot
     * counter mask, so the scanner's positional read read_hpmcounter(3+j)
     * always targets the event it opened — no dependence on OpenSBI's
     * free-counter search order.  This is what lets the speculative scanner
     * keep three fixed filler events on hpm4/5/6 and rotate only the event
     * under test on hpm3.  Only pin onto counters the board actually exposes
     * (those present in g_hpm_scan_mask); anything past that falls back to the
     * wide mask.  pmu_do_close_raw() above guarantees the counters are free. */

    /* ── Phase 1: DTB-validated allocation + explicit start ──
     *
     * AUTO_START|CLEAR_VALUE (PMU_CFG_FLAG_OPEN) was tried but OpenSBI 1.4
     * on this board accepts the flag without actually clearing mcountinhibit.
     * The counter is configured (mhpmeventX written) but stays inhibited, so
     * every delta reads 0.  Use flags=0 in CFG_MATCH, then an explicit START
     * call with INIT_VALUE=0, which is confirmed working.
     */
    for (i = 0; i < n; i++) {
        unsigned long want = BIT(3 + i);
        unsigned long ev_mask = (g_hpm_scan_mask & want) ? want : g_hpm_scan_mask;

        struct sbiret ret = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_CFG_MATCH,
                                      0,
                                      ev_mask,
                                      0,   /* no flags — AUTO_START broken in firmware */
                                      SBI_PMU_RAW_EVENT_IDX,
                                      req->event_data[i],
                                      0);
        if (ret.error)
            continue;

        req->ctr_idx[i] = (int)ret.value;

        if ((g_hpm_scan_mask & want) && req->ctr_idx[i] != (int)(3 + i))
            pr_warn("PMU: slot %d pin expected hpm%d, got hpm%d\n",
                    i, 3 + i, req->ctr_idx[i]);

        /* Explicit start with counter zeroed */
        ret = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_START,
                        0, 1UL << req->ctr_idx[i],
                        PMU_START_SET_INIT_VALUE, 0, 0, 0);

        if (ret.error && ret.error != SBI_ERR_ALREADY_STARTED) {
            sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_STOP,
                      req->ctr_idx[i], 0x1, PMU_STOP_FLAG_RESET, 0, 0, 0);
            req->ctr_idx[i] = -1;
            continue;
        }

        req->ok[i] = 1;
        g_raw_ctr_mask |= (1UL << req->ctr_idx[i]);
        n_phase1++;
    }

    pr_info("PMU: OPEN_RAW phase1 (DTB): %d events opened, mask=0x%08x\n",
            n_phase1, g_raw_ctr_mask);

    /* ── Phase 2: SKIP_MATCH on individual counters beyond g_hpm_scan_mask ── */
    for (i = 0; i < n; i++) {
        if (req->ok[i])
            continue;   /* already opened in phase 1 */

        /* Try each counter index 4–31 with a single-bit mask + SKIP_MATCH.
         * Single-bit mask avoids OpenSBI rejecting the call due to out-of-range
         * bits; if counter j doesn't exist, the call fails cleanly for BIT(j). */
        for (int j = 4; j <= 31; j++) {
            if (g_raw_ctr_mask & BIT(j))
                continue;   /* already allocated */
            if (g_hpm_scan_mask & BIT(j))
                continue;   /* in DTB set, already tried in phase 1 */

            struct sbiret ret = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_CFG_MATCH,
                                          0,
                                          BIT(j),
                                          0,   /* no AUTO_START — broken in firmware */
                                          SBI_PMU_RAW_EVENT_IDX,
                                          req->event_data[i],
                                          0);
            if (ret.error)
                continue;

            req->ctr_idx[i] = (int)ret.value;

            ret = sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_START,
                            0, 1UL << req->ctr_idx[i],
                            PMU_START_SET_INIT_VALUE, 0, 0, 0);

            if (ret.error && ret.error != SBI_ERR_ALREADY_STARTED) {
                sbi_ecall(SBI_EXT_PMU, SBI_EXT_PMU_FID_STOP,
                          req->ctr_idx[i], 0x1, PMU_STOP_FLAG_RESET, 0, 0, 0);
                req->ctr_idx[i] = -1;
                continue;
            }

            req->ok[i] = 1;
            g_raw_ctr_mask |= (1UL << req->ctr_idx[i]);
            n_phase2++;
            pr_info("PMU:   SKIP_MATCH: event 0x%llx → hpmcounter%d\n",
                    req->event_data[i], req->ctr_idx[i]);
            break;   /* found a counter for this event, move to next event */
        }
    }

    pr_info("PMU: OPEN_RAW total: phase1=%d phase2(SKIP_MATCH)=%d, "
            "active_mask=0x%08x\n",
            n_phase1, n_phase2, g_raw_ctr_mask);
}

/* =========================================================
 * IOCTL_PMU_TEARDOWN — release all counters and perf events
 * ========================================================= */
static void pmu_do_teardown(void)
{
    int i;

    pmu_do_close_raw();

    for (i = 0; i < 3; i++) {
        if (g_perf_ev[i] && !IS_ERR(g_perf_ev[i])) {
            perf_event_release_kernel(g_perf_ev[i]);
            g_perf_ev[i] = NULL;
        }
    }

    pmu_stop((1UL << 0) | (1UL << 2));

    pr_info("PMU: teardown complete\n");
}

/* =========================================================
 * IOCTL dispatch
 * ========================================================= */
static long device_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
    switch (cmd) {

    case IOCTL_PMU_SETUP: {
        struct pmu_setup_info info;
        if (copy_from_user(&info, (void __user *)arg, sizeof(info)))
            return -EFAULT;
        pmu_do_setup(&info);
        if (copy_to_user((void __user *)arg, &info, sizeof(info)))
            return -EFAULT;
        return 0;
    }

    case IOCTL_PMU_TEARDOWN:
        pmu_do_teardown();
        return 0;

    case IOCTL_PMU_RAW_SCAN: {
        struct pmu_raw_scan_result result;
        pmu_do_raw_scan(&result);
        if (copy_to_user((void __user *)arg, &result, sizeof(result)))
            return -EFAULT;
        return 0;
    }

    case IOCTL_PMU_OPEN_RAW: {
        struct pmu_open_raw_request req;
        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;
        pmu_do_open_raw(&req);
        if (copy_to_user((void __user *)arg, &req, sizeof(req)))
            return -EFAULT;
        return 0;
    }

    case IOCTL_PMU_CLOSE_RAW:
        pmu_do_close_raw();
        return 0;

    default:
        return -EINVAL;
    }
}

/* =========================================================
 * Character device boilerplate
 * ========================================================= */
static int device_open(struct inode *i, struct file *f)    { return 0; }
static int device_release(struct inode *i, struct file *f) { return 0; }

static const struct file_operations fops = {
    .open           = device_open,
    .release        = device_release,
    .unlocked_ioctl = device_ioctl,
};

static struct miscdevice pmu_misc = {
    .minor = MISC_DYNAMIC_MINOR,
    .name  = DEVICE_NAME,
    .fops  = &fops,
};

/* =========================================================
 * Module init / exit
 * ========================================================= */
static int __init pmu_init(void)
{
    int ret = misc_register(&pmu_misc);
    if (ret) {
        pr_err("PMU: misc_register failed: %d\n", ret);
        return ret;
    }
    pr_info("PMU: /dev/%s ready\n", DEVICE_NAME);
    return 0;
}

static void __exit pmu_exit(void)
{
    pmu_do_teardown();
    misc_deregister(&pmu_misc);
    pr_info("PMU: unloaded\n");
}

module_init(pmu_init);
module_exit(pmu_exit);