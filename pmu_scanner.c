/* pmu_scanner.c - RISC-V opcode scanner for EIC7700X (ESWIN 770x) */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>


// Majority vote settings
#define SETTLE_WINDOW_DEFAULT 16   /* default settle-window size; override with --window N */

// Adding NOP sled after alignment helps tweak speculative window reliability and performance. Can be set from compiler too
#ifndef SPEC_GADGET_RUNWAY_NOPS
#define SPEC_GADGET_RUNWAY_NOPS 20
#endif
#define SPEC_STR_(x)  #x
#define SPEC_XSTR_(x) SPEC_STR_(x)

#ifndef TARGET_MILKV_PIONEER
#define CPU_NUMBER 3
#else
#define CPU_NUMBER 63
#endif


// Kernel module interface (matches pmu_module.c)
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

#define IOCTL_PMU_SETUP    _IOWR('p', 5, struct pmu_setup_info)
#define IOCTL_PMU_TEARDOWN _IO  ('p', 6)

// pmu_module globals
#define PMU_RAW_MAX_EVENTS  512   /* scan table capacity */
#define PMU_RAW_CTR_MAX       4   /* hardware HPM counters: hpm3–hpm6 */


struct pmu_raw_event {
    uint64_t event_data;  /* raw mhpmevent value accepted by firmware */
    int      ctr_idx;     /* hardware counter index (3-31) */
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

/* Settle-window ring buffers (heap-allocated, runtime-sized). */
static int        g_settle_window;     /* active window size */
static int        g_settle_majority;   /* n/2 + 1; unique-mode threshold */
static uint64_t  *g_w_ir;              /* [g_settle_window] instret ring */
static uint64_t  *g_w_cy;              /* [g_settle_window] cycle ring */
static uint64_t (*g_w_hpm)[PMU_RAW_CTR_MAX];  /* [g_settle_window][ctr] hpm ring */
static uint64_t  *g_col;               /* [g_settle_window] per-column vote scratch */

/* Verify-mode analysis windows (post-hoc), populated from --windows. */
#define MAX_ANALYSIS_WINDOWS 16
static int        g_analysis_windows[MAX_ANALYSIS_WINDOWS];
static int        g_n_analysis_windows;

// Allocate the settle-window ring buffers for a given window size. The window is
static int init_settle_window(int n)
{
    if (n < 1) {
        fprintf(stderr, "  settle window must be >= 1 (got %d)\n", n);
        return -1;
    }
    g_settle_window   = n;
    g_settle_majority = n / 2 + 1;

    g_w_ir  = malloc((size_t)n * sizeof *g_w_ir);
    g_w_cy  = malloc((size_t)n * sizeof *g_w_cy);
    g_w_hpm = malloc((size_t)n * sizeof *g_w_hpm);
    g_col   = malloc((size_t)n * sizeof *g_col);
    if (!g_w_ir || !g_w_cy || !g_w_hpm || !g_col) {
        perror("  malloc settle buffers");
        return -1;
    }
    return 0;
}

#define IOCTL_PMU_RAW_SCAN  _IOR ('p', 7, struct pmu_raw_scan_result)
#define IOCTL_PMU_OPEN_RAW  _IOWR('p', 8, struct pmu_open_raw_request)
#define IOCTL_PMU_CLOSE_RAW _IO  ('p', 9)

// Read CSR macro
#define READ_CSR(csr_num, dst) \
    asm volatile ("csrr %0, " #csr_num : "=r"(dst) :: "memory")

// Variable to hold the RSB gadget
static uint32_t *g_exec_page;

// RSB Gadget layout
#define GADGET_SAVE_RA     0   /* addi t0, ra, 0      - Save return address */
#define GADGET_JAL         1   /* jal  ra, +28        - acts as call instruction to slot 8 */
#define GADGET_CANDIDATE   2   /* <candidate>         - Call would return here, but architecturally we changed that */
#define GADGET_NOP1        3   /* nop */
#define GADGET_NOP2        4   /* nop */
#define GADGET_RESTORE_RA  5   /* addi ra, t0, 0      - safe_return entry */
#define GADGET_RET         6   /* ret */
#define GADGET_PAD         7   /* nop                 - alignment pad */
#define GADGET_CALLEE_LD   8   /* ld   ra, 0(a0)      - Load architectural return address, this slow load forces RSB speculation */
#define GADGET_CALLEE_RET  9   /* ret                 - RSB misprediction */

// Risc-V instruction encodings for the gadget
#define ENC_NOP             0x00000013U     /* addi x0, x0, 0 */
#define ENC_RET             0x00008067U     /* jalr x0, 0(ra) */
#define ENC_ADDI_T0_RA_0    0x00008293U     /* addi t0, ra, 0 */
#define ENC_JAL_RA_P28      0x01C000EFU     /* jal  ra, +28 */
#define ENC_ADDI_RA_T0_0    0x00028093U     /* addi ra, t0, 0 */
#define ENC_LD_RA_A0        0x00053083U     /* ld   ra, 0(a0) */
#define ENC_LR_D_RA_A0      0x000530af     /* lr.d ra, (a0) */

// Page size helpers
#define PAGE_SIZE           4096
#define HUGEPAGE_SIZE       (2 * 1024 * 1024)


// L1D geometry per-target; derive sets and way-stride from size/ways/line.
#if defined(TARGET_MILKV_PIONEER)
#define L1D_TOTAL_SIZE      (64 * 1024)   /* thead,c920 d-cache-size */
#define L1D_WAYS            2             /* derived, see comment above */
#define L1D_LINE_SIZE       64            /* thead,c920 d-cache-block-size */
#else /* default: HiFive Premier P550 */
#define L1D_TOTAL_SIZE      (32 * 1024)   /* SiFive P550 documentation */
#define L1D_WAYS            4             /* SiFive P550 documentation */
#define L1D_LINE_SIZE       64            /* SiFive P550 documentation */
#endif

// Derived, not sourced - see comment above.
#define L1D_SETS        (L1D_TOTAL_SIZE / (L1D_WAYS * L1D_LINE_SIZE))
#define L1D_WAY_STRIDE  (L1D_SETS * L1D_LINE_SIZE)

#define SLOW_TARGET_OFFSET 0

#define EVICT_STRIDE     L1D_WAY_STRIDE   /* same set, next line - board-derived */
// Passes per eviction call. 4 was tuned empirically against the P550's
#define EVICT_PASSES     4



#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

/* Huge-page arena for slow target and probe line (2MB huge page, VA==PA low bits). */
static uint8_t         *g_evict_arena;   /* single 2MB huge page (MAP_HUGETLB) */

// RSB gadget return address placed at arena offset 0 (congruent evict).
#define g_slow_target_ptr ((volatile uint64_t *)(g_evict_arena + SLOW_TARGET_OFFSET))
#define g_slow_target      (*g_slow_target_ptr)

/* ── Window-quality probe (verification harness) ──────────────────────────── */
/* A dedicated probe line reached by the candidate as ld t1, PROBE_LINE_OFFSET(a0). */
#define PROBE_LINE_OFFSET   1984                       /* 31*64: set 31, ld-imm reach */
static volatile uint64_t *g_probe_line_ptr;      /* arena + PROBE_LINE_OFFSET */

/* Deep flush: a huge-page-backed buffer walked to capacity-evict the probe line */
#if defined(TARGET_MILKV_PIONEER)
#define DEEP_FLUSH_SIZE (64 * HUGEPAGE_SIZE)    /* 128MB = 2x the 64MB L3 */
#else /* default: HiFive Premier P550 */
#define DEEP_FLUSH_SIZE (2 * HUGEPAGE_SIZE)     /* 4MB = 2 huge pages */
#endif
static uint8_t *g_deep_flush_buf;

/* Targeted L2 eviction of the slow target. */
#if defined(TARGET_MILKV_PIONEER)
#define L2_TOTAL_SIZE    (1024 * 1024)    /* Milk-V docs: 1MB per cluster */
#define L2_WAYS          16               /* inferred, see comment above - LOW CONFIDENCE */
#define L2_LINE_SIZE     64               /* assumed, matches L1D block size */
#define L2_EVICT_LINES   32                /* 2x L2_WAYS - kept modest: see DEEP_FLUSH_SIZE bound below */
#else /* default: HiFive Premier P550 */
#define L2_TOTAL_SIZE    (256 * 1024)      /* SiFive P550 documentation */
#define L2_WAYS          8                 /* SiFive P550 documentation */
#define L2_LINE_SIZE     64                /* SiFive P550 documentation */
#define L2_EVICT_LINES   32                /* >> 8-way, margin for RRIP */
#endif
// Derived, not sourced - see comment above.
#define L2_EVICT_STRIDE  (L2_TOTAL_SIZE / L2_WAYS)
#define L2_EVICT_PASSES  3

/* The L2 eviction chain lives inside g_deep_flush_buf at offsets */
_Static_assert((size_t)(L2_EVICT_LINES - 1) * L2_EVICT_STRIDE < DEEP_FLUSH_SIZE,
               "L2 eviction chain overruns g_deep_flush_buf - "
               "reduce L2_EVICT_LINES or grow DEEP_FLUSH_SIZE");

static void * volatile *g_l2_evict_head;
static volatile uintptr_t g_l2_evict_sink;


// Init executable page and g_slow_target, set up RSB gadget
static void init_exec_page(void)
{
// Two pages: page 0 holds the gadget; page 1 is left zero-filled
    g_exec_page = mmap(NULL, PAGE_SIZE * 2,
                       PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g_exec_page == MAP_FAILED) {
        perror("mmap exec page");
        exit(1);
    }

// Write fixed gadget words.  Only GADGET_CANDIDATE is updated per probe.
    g_exec_page[GADGET_SAVE_RA]    = ENC_ADDI_T0_RA_0;
    g_exec_page[GADGET_JAL]        = ENC_JAL_RA_P28;
    g_exec_page[GADGET_CANDIDATE]  = ENC_NOP;           /* placeholder */
    g_exec_page[GADGET_NOP1]       = ENC_NOP;
    g_exec_page[GADGET_NOP2]       = ENC_NOP;
    g_exec_page[GADGET_RESTORE_RA] = ENC_ADDI_RA_T0_0;
    g_exec_page[GADGET_RET]        = ENC_RET;
    g_exec_page[GADGET_PAD]        = ENC_NOP;
    g_exec_page[GADGET_CALLEE_LD]  = ENC_LD_RA_A0;
    g_exec_page[GADGET_CALLEE_RET] = ENC_RET;
// Flush I-cache for the entire gadget region
    __builtin___clear_cache((char *)g_exec_page,
                            (char *)(g_exec_page + GADGET_CALLEE_RET + 1));

// Store architectural return address
    g_slow_target = (uint64_t)(uintptr_t)&g_exec_page[GADGET_RESTORE_RA];
}

static void destroy_exec_page(void)
{
    if (g_exec_page) {
        munmap(g_exec_page, PAGE_SIZE * 2);
        g_exec_page = NULL;
    }
}


// Flush the slow target address to ensure it is not in cache before the probe
#define EVICT_SIZE  (512 * 1024)
static __attribute__((aligned(PAGE_SIZE))) volatile uint8_t g_evict_buf[EVICT_SIZE];

// Ensure eviction buffer is not backed by kernel zero pages.
static void init_evict_buf(void)
{
    memset((void *)g_evict_buf, 0xA5, EVICT_SIZE);
}

static void flush_slow_target(void)
{
    volatile uint8_t *p = g_evict_buf;
    volatile uint8_t sink = 0;
    for (size_t i = 0; i < EVICT_SIZE; i += 64)
        sink ^= p[i];
    (void)sink;

// Full fence to ensure all memory operations are completed before proceeding
    asm volatile ("fence rw, rw" ::: "memory");
}


// mmap the arena as a single 2MB huge page, nonzero-init it, and build the
static void init_evict_arena(void)
{
    g_evict_arena = mmap(NULL, HUGEPAGE_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,
                         -1, 0);
    if (g_evict_arena == MAP_FAILED) {
        fprintf(stderr,
                "mmap eviction arena (MAP_HUGETLB) failed: %s\n"
                "  huge pages must be reserved first (this build needs %zu "
                "total: 1 for the arena + %zu for the deep-flush buffer), "
                "e.g.:\n"
                "    sudo sysctl vm.nr_hugepages=%zu\n",
                strerror(errno),
                (size_t)(1 + DEEP_FLUSH_SIZE / HUGEPAGE_SIZE),
                (size_t)(DEEP_FLUSH_SIZE / HUGEPAGE_SIZE),
                (size_t)(1 + DEEP_FLUSH_SIZE / HUGEPAGE_SIZE));
        exit(1);
    }
    /* Nonzero-init the page so it is backed by real physical memory and starts */
    memset(g_evict_arena, 0xA5, HUGEPAGE_SIZE);
}


/* ── Probe-line eviction (mirror of the slow-target chain, set 31) ─────────── */

// Must run AFTER init_evict_arena() (needs g_evict_arena). Writes a known value
static void init_probe_line(void)
{
    g_probe_line_ptr = (volatile uint64_t *)(g_evict_arena + PROBE_LINE_OFFSET);
    *g_probe_line_ptr = 0xC0FFEEULL;
}

// Allocate the deep-flush buffer (call once at setup). Huge-page-backed for
static void init_deep_flush(void)
{
    g_deep_flush_buf = mmap(NULL, DEEP_FLUSH_SIZE, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,
                            -1, 0);
    if (g_deep_flush_buf == MAP_FAILED) {
        fprintf(stderr,
                "mmap flush/L2-evict buffer (MAP_HUGETLB) failed: %s\n"
                "  %zu MB (%zu huge pages) needed for this buffer alone; "
                "this build needs %zu total huge pages reserved "
                "(arena + deep-flush buffer), e.g.:\n"
                "    sudo sysctl vm.nr_hugepages=%zu\n",
                strerror(errno),
                (size_t)(DEEP_FLUSH_SIZE >> 20),
                (size_t)(DEEP_FLUSH_SIZE / HUGEPAGE_SIZE),
                (size_t)(1 + DEEP_FLUSH_SIZE / HUGEPAGE_SIZE),
                (size_t)(1 + DEEP_FLUSH_SIZE / HUGEPAGE_SIZE));
        exit(1);
    }
    memset(g_deep_flush_buf, 0xA5, DEEP_FLUSH_SIZE);

    /* Build the targeted L2 eviction chain (32KB-strided, L2-set-0 congruent). */
    void * volatile *prev = NULL;
    for (int i = L2_EVICT_LINES - 1; i >= 0; i--) {
        void * volatile *slot = (void * volatile *)
            (g_deep_flush_buf + (size_t)i * L2_EVICT_STRIDE);
        *slot = (void *)prev;
        prev  = slot;
    }
    g_l2_evict_head = prev;
}

// Pointer-chase the slow target's L2-congruent set, several passes, to force out
static inline void evict_slow_target_l2(void)
{
    for (int p = 0; p < L2_EVICT_PASSES; p++) {
        void * volatile *q = g_l2_evict_head;
        while (q) q = (void * volatile *)*q;
        g_l2_evict_sink = (uintptr_t)q;
    }
    asm volatile ("fence rw, rw" ::: "memory");
}

// Capacity-evict the probe line and slow target to DRAM depth. One pass over a
static inline void deep_flush(void)
{
    volatile uint8_t s = 0;
    for (size_t o = 0; o < DEEP_FLUSH_SIZE; o += 64) s ^= g_deep_flush_buf[o];
    (void)s;
    asm volatile ("fence rw, rw" ::: "memory");
}


// Probe result struct
typedef struct {
    uint32_t encoding;

    /* Fixed counter deltas across the probe window.  In speculative mode these */
    uint64_t cycle_delta;
    uint64_t instret_delta;
} probe_result_t;


// Find the majority value in an array of uint64_t values. Returns its count and sets actual value in out_val
static int majority_val(const uint64_t *w, int n, uint64_t *out_val)
{
    int best = 0;
    uint64_t best_v = 0;
    for (int a = 0; a < n; a++) {
        int cnt = 0;
        for (int b = 0; b < n; b++)
            if (w[b] == w[a]) cnt++;
        if (cnt > best) { best = cnt; best_v = w[a]; }
    }
    *out_val = best_v;
    return best;
}


// Min/max/mode summary of a set of latency samples. Mode reuses majority_val
typedef struct {
    uint64_t min;
    uint64_t max;
    uint64_t avg;
    uint64_t mode;
} latency_stats_t;

static latency_stats_t latency_stats(const uint64_t *samples, int n)
{
    latency_stats_t st;
    uint64_t sum = 0;

    st.min = samples[0];
    st.max = samples[0];
    sum = samples[0];
    for (int i = 1; i < n; i++) {
        if (samples[i] < st.min) st.min = samples[i];
        if (samples[i] > st.max) st.max = samples[i];
        sum += samples[i];
    }
    st.avg = sum / (uint64_t)n;
    majority_val(samples, n, &st.mode);
    return st;
}

// Times a single load of g_slow_target, bracketed tightly by rdcycle reads.
static inline __attribute__((always_inline))
uint64_t time_slow_target_load(void)
{
    uint64_t t0, t1;
    READ_CSR(0xc00, t0);
    volatile uint64_t sink = g_slow_target;
    READ_CSR(0xc00, t1);
    (void)sink;
    return t1 - t0;
}

// Times one architectural load of `addr`, with both ends pinned against
static inline __attribute__((always_inline))
uint64_t time_addr_serialized(volatile uint64_t *addr)
{
    uint64_t t0, t1, val, z;
    uint64_t a = (uint64_t)(uintptr_t)addr;
    asm volatile(
        "fence   rw, rw            \n\t"
        "rdcycle %[t0]             \n\t"
        "sub     %[z], %[t0], %[t0]\n\t"   /* z = 0, depends on t0 */
        "add     %[a], %[a], %[z]  \n\t"   /* addr now depends on t0 */
        "ld      %[v], 0(%[a])     \n\t"   /* load cannot issue before t0 */
        "fence   rw, rw            \n\t"
        "fence.i                   \n\t"   /* drain: miss retires before t1 */
        "rdcycle %[t1]             \n\t"
        : [t0]"=&r"(t0), [t1]"=&r"(t1), [v]"=&r"(val), [z]"=&r"(z), [a]"+&r"(a)
        :
        : "memory");
    return t1 - t0;
}

// Times one architectural load of the probe line (arena + PROBE_LINE_OFFSET).
static inline __attribute__((always_inline))
uint64_t time_probe_line_load(void)
{
    return time_addr_serialized(g_probe_line_ptr);
}

// Encode  ld rd, imm(rs1)  : I-type, funct3=0b011 (LD), opcode=0b0000011.
static inline uint32_t enc_ld(int rd, int rs1, int imm12)
{
    return (((uint32_t)imm12 & 0xFFF) << 20) | (((uint32_t)rs1 & 0x1F) << 15)
         | (0x3u << 12) | (((uint32_t)rd & 0x1F) << 7) | 0x03u;
}

#define EVICT_TEST_TRIALS 3000

// Auto-ranging histogram of a sample array. Width-1 buckets across the first
#define HIST_FINE 32
static void print_histogram(const char *label, const uint64_t *s, int n)
{
    static const uint64_t cw[4] = {16, 32, 64, 128};  /* coarse widths: +48,+80,+144,+272 */

    uint64_t mn = s[0];
    for (int i = 1; i < n; i++) if (s[i] < mn) mn = s[i];

    uint64_t fine[HIST_FINE] = {0};
    uint64_t coarse[5]       = {0};

    for (int i = 0; i < n; i++) {
        uint64_t d = s[i] - mn;
        if (d < (uint64_t)HIST_FINE) { fine[d]++; continue; }
        uint64_t lo = HIST_FINE; int placed = 0;
        for (int c = 0; c < 4; c++) {
            if (d < lo + cw[c]) { coarse[c]++; placed = 1; break; }
            lo += cw[c];
        }
        if (!placed) coarse[4]++;
    }

    uint64_t peak = 1;
    for (int b = 0; b < HIST_FINE; b++) if (fine[b]   > peak) peak = fine[b];
    for (int b = 0; b < 5;         b++) if (coarse[b] > peak) peak = coarse[b];

    printf("    %s histogram (cycles, min=%lu):\n", label, (unsigned long)mn);
    for (int b = 0; b < HIST_FINE; b++) {
        if (fine[b] == 0) continue;                    /* suppress empty fine rows */
        int bar = (int)((fine[b] * 40) / peak);
        printf("      %-9lu %6lu  %5.1f%%  %.*s\n",
               (unsigned long)(mn + b), (unsigned long)fine[b],
               100.0 * (double)fine[b] / (double)n,
               bar, "########################################");
    }
    uint64_t lo = mn + HIST_FINE;
    for (int c = 0; c < 5; c++) {
        char range[24];
        if (c < 4) {
            snprintf(range, sizeof range, "%lu-%lu",
                     (unsigned long)lo, (unsigned long)(lo + cw[c] - 1));
            lo += cw[c];
        } else {
            snprintf(range, sizeof range, ">=%lu", (unsigned long)lo);
        }
        int bar = (int)((coarse[c] * 40) / peak);
        printf("      %-9s %6lu  %5.1f%%  %.*s\n",
               range, (unsigned long)coarse[c],
               100.0 * (double)coarse[c] / (double)n,
               bar, "########################################");
    }
}

// Stores 1 sample of raw counter deltas for a single probe
typedef struct {
    uint64_t cycle;
    uint64_t instret;
    uint64_t hpm[PMU_RAW_CTR_MAX];   /* [0]=hpm3, [1]=hpm4, [2]=hpm5, [3]=hpm6 */
} raw_sample_t;

// Perform one RSB gadget execution and store raw counter deltas.
static inline __attribute__((always_inline))
raw_sample_t probe_spec_single(const struct pmu_open_raw_request *breq, int bsz)
{
    (void)breq; (void)bsz;

    raw_sample_t s;
    uint64_t cy_b, ir_b, h3_b, cy_a, ir_a, h3_a;

    asm volatile(".p2align 7");
    READ_CSR(0xc00, cy_b);
    READ_CSR(0xc02, ir_b);
    READ_CSR(0xc03, h3_b);              /* hpm3, unconditional bare csrr */

    asm volatile(".rept " SPEC_XSTR_(SPEC_GADGET_RUNWAY_NOPS) "\n\t"
                 "nop\n\t"
                 ".endr");
    ((void (*)(volatile uint64_t *))g_exec_page)(&g_slow_target);

    READ_CSR(0xc00, cy_a);
    READ_CSR(0xc02, ir_a);
    READ_CSR(0xc03, h3_a);

    s.cycle   = cy_a - cy_b;
    s.instret = ir_a - ir_b;
    s.hpm[0]  = h3_a - h3_b;
    s.hpm[1]  = 0;
    s.hpm[2]  = 0;
    s.hpm[3]  = 0;

    return s;
}

// Diagnostic-only: pure slow-target eviction quality (see below).
static void test_eviction_quality(int fd, int cpu)
{
    (void)fd; (void)cpu;
    uint64_t warm[EVICT_TEST_TRIALS];
    uint64_t evicted[EVICT_TEST_TRIALS];

    /* Measure the slow target's own load latency, fenced, warm vs evicted. This */
    for (int i = 0; i < EVICT_TEST_TRIALS; i++) {
        volatile uint64_t touch;

        touch = g_slow_target; (void)touch;                  /* warm */
        warm[i] = time_addr_serialized(g_slow_target_ptr);

        touch = g_slow_target; (void)touch;
        evict_slow_target_l2();                              /* targeted L2 evict */
        evicted[i] = time_addr_serialized(g_slow_target_ptr);
    }

    latency_stats_t w = latency_stats(warm, EVICT_TEST_TRIALS);
    latency_stats_t e = latency_stats(evicted, EVICT_TEST_TRIALS);

    printf("\n  Slow-target load latency (%d trials each, warm vs L2-evicted):\n",
           EVICT_TEST_TRIALS);
        printf("    WARM     min=%-6lu max=%-6lu avg=%-6lu mode=%-6lu\n",
            (unsigned long)w.min, (unsigned long)w.max,
            (unsigned long)w.avg, (unsigned long)w.mode);
        printf("    EVICTED  min=%-6lu max=%-6lu avg=%-6lu mode=%-6lu\n",
            (unsigned long)e.min, (unsigned long)e.max,
            (unsigned long)e.avg, (unsigned long)e.mode);
    if (w.min > 0)
        printf("    ratio (evicted.min / warm.min): %.2f\n",
               (double)e.min / (double)w.min);

    print_histogram("WARM   ", warm,    EVICT_TEST_TRIALS);
    print_histogram("EVICTED", evicted, EVICT_TEST_TRIALS);

    printf("    (diagnostic only - does not affect scan execution)\n\n");
}


// Diagnostic-only: does the RSB speculation window actually open, and how often?
#define WINDOW_TEST_TRIALS   300
#define WINDOW_CALIB_TRIALS  100

static void test_window_quality(int fd, int cpu)
{
    if (!g_probe_line_ptr) {
        fprintf(stderr, "  window diagnostic: probe line not initialised - "
                        "init_probe_line() missing or ran before init_evict_arena() "
                        "(skipping)\n");
        return;
    }

    /* Same batch shape as the scan: a single counter (the event-under-test) on */
    struct pmu_open_raw_request breq;
    memset(&breq, 0, sizeof(breq));
    breq.cpu   = cpu;
    breq.count = 1;
    breq.event_data[0] = 0x2003;

    flush_slow_target();
    if (ioctl(fd, IOCTL_PMU_OPEN_RAW, &breq) != 0) {
        fprintf(stderr, "  window diagnostic: IOCTL_PMU_OPEN_RAW failed: %s"
                        " (skipping)\n", strerror(errno));
        return;
    }
    const int bsz = breq.count;

    /* 1. Calibrate the cached/uncached threshold on the probe line itself. */
    static uint64_t warm[WINDOW_CALIB_TRIALS];
    static uint64_t cold[WINDOW_CALIB_TRIALS];
    for (int i = 0; i < WINDOW_CALIB_TRIALS; i++) {
        volatile uint64_t t = *g_probe_line_ptr; (void)t;   /* warm it */
        warm[i] = time_probe_line_load();
        deep_flush();
        cold[i] = time_probe_line_load();
    }
    latency_stats_t ws = latency_stats(warm, WINDOW_CALIB_TRIALS);
    latency_stats_t cs = latency_stats(cold, WINDOW_CALIB_TRIALS);
    /* Anchor the threshold to the stable warm/fill peak, not the deep_flush cold */
    uint64_t threshold = ws.mode + ws.mode / 3;

    printf("\n  Window-quality probe (probe line @ arena+%d, set %d):\n",
           PROBE_LINE_OFFSET, (PROBE_LINE_OFFSET >> 6) & (L1D_SETS - 1));
    printf("    calib: warm mode=%lu  evicted mode=%lu  threshold=%lu\n",
           (unsigned long)ws.mode, (unsigned long)cs.mode,
           (unsigned long)threshold);
    if (cs.mode <= threshold) {
        printf("    WARNING: deep_flush cold mode (%lu) not above threshold (%lu) "
               "- not-filled state indistinguishable from a fill, skipping\n",
               (unsigned long)cs.mode, (unsigned long)threshold);
        ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);
        return;
    }

    /* 1b. Confirm the deep flush actually evicts the slow target (arena+0) */
    static uint64_t stw[WINDOW_CALIB_TRIALS];
    static uint64_t stc[WINDOW_CALIB_TRIALS];
    for (int i = 0; i < WINDOW_CALIB_TRIALS; i++) {
        volatile uint64_t t = g_slow_target; (void)t;        /* warm slow target */
        stw[i] = time_addr_serialized(g_slow_target_ptr);
        deep_flush();
        evict_slow_target_l2();                              /* targeted L2 evict */
        stc[i] = time_addr_serialized(g_slow_target_ptr);
    }
    latency_stats_t sw = latency_stats(stw, WINDOW_CALIB_TRIALS);
    latency_stats_t sc = latency_stats(stc, WINDOW_CALIB_TRIALS);
    printf("    slow-target: warm mode=%lu  flushed mode=%lu  "
           "(window opens iff flushed >> warm)\n",
           (unsigned long)sw.mode, (unsigned long)sc.mode);
    /* Distribution, not just the mode: a warm tail here = trials where the L2 */
    print_histogram("SLOW-FLUSHED", stc, WINDOW_CALIB_TRIALS);



    /* 2. Two passes: candidate = load-of-probe (via a0, the real gadget's arg), */
    const uint32_t cands[2]  = { enc_ld(/* rd=t1 */6, /* rs1=a0 */10, PROBE_LINE_OFFSET),
                                 ENC_NOP };
    const char    *labels[2] = { "ld t1,off(a0)", "nop (control)" };
    long hits[2] = {0, 0};

    for (int pass = 0; pass < 2; pass++) {
        g_exec_page[GADGET_CANDIDATE] = cands[pass];
        __builtin___clear_cache((char *)g_exec_page,
                                (char *)(g_exec_page + GADGET_CALLEE_RET + 1));

        for (int i = 0; i < WINDOW_TEST_TRIALS; i++) {
            deep_flush();                          /* probe line (set 31) -> DRAM */
            evict_slow_target_l2();                /* slow target (set 0) -> L3 (window) */
            (void)probe_spec_single(&breq, bsz);
            if (time_probe_line_load() < threshold) /* transient candidate filled it */
                hits[pass]++;
        }
        printf("    %-16s cached %ld / %d  (%.2f%%)\n",
               labels[pass], hits[pass], WINDOW_TEST_TRIALS,
               100.0 * (double)hits[pass] / (double)WINDOW_TEST_TRIALS);
    }

    printf("    transient-fire rate (load - nop, window confirmed open above): %.2f%%\n",
           100.0 * (double)(hits[0] - hits[1]) / (double)WINDOW_TEST_TRIALS);

    /* Restore a clean gadget for the subsequent scan. */
    g_exec_page[GADGET_CANDIDATE] = ENC_NOP;
    __builtin___clear_cache((char *)g_exec_page,
                            (char *)(g_exec_page + GADGET_CALLEE_RET + 1));

    ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);
    printf("    (diagnostic only - does not affect scan execution)\n\n");
}


// Perform a batch of speculative probes without flushing the instruction cache. Returns the majority values of the counters across the batch.
 __attribute__((noinline))
static void probe_spec_batch_no_flush(uint32_t encoding,
                                      const struct pmu_open_raw_request *breq,
                                      int bsz, int bs,
                                      int first_batch,
                                      uint64_t *out_cy, uint64_t *out_ir,
                                      uint64_t raw_row[PMU_RAW_MAX_EVENTS],
                                      int *out_settled)
{
    /* Ring buffers live on the heap (allocated once by init_settle_window) */
    uint64_t                *w_ir  = g_w_ir;
    uint64_t                *w_cy  = g_w_cy;
    uint64_t               (*w_hpm)[PMU_RAW_CTR_MAX] = g_w_hpm;
    const int                wcap  = g_settle_window;
    const int                wmaj  = g_settle_majority;
    int      whead = 0, wfull = 0;

// If fallback to mean is used, out_settled is set to 0. Otherwise, it is set to 1. Used by verify mode.
    if (out_settled) *out_settled = 1;

    memset(w_ir,  0, (size_t)wcap * sizeof *w_ir);
    memset(w_cy,  0, (size_t)wcap * sizeof *w_cy);
    memset(w_hpm, 0, (size_t)wcap * sizeof *w_hpm);

    g_exec_page[GADGET_CANDIDATE] = encoding;
    __builtin___clear_cache((char *)&g_exec_page[GADGET_CANDIDATE],
                            (char *)(&g_exec_page[GADGET_CANDIDATE] + 1));

    uint64_t touch;
    int attempts = 0;
    for (int iter = 0; iter < wcap; iter++) {
        touch = g_slow_target; (void)touch;
        evict_slow_target_l2();   /* targeted L2 (+L1) evict: opens the window reliably */
        raw_sample_t sample = probe_spec_single(breq, bsz);

        w_ir [whead] = sample.instret;
        w_cy [whead] = sample.cycle;
        for (int j = 0; j < bsz; j++)
            w_hpm[whead][j] = breq->ok[j] ? sample.hpm[j] : 0;
        if (++whead == wcap) whead = 0;   /* branch wrap: value-independent codegen */
        if (!wfull && whead == 0) wfull = 1;

        int wsz = wfull ? wcap : whead;
        if (wsz < wcap)
            continue;

// Joint majority vote across all active metrics and counters (mcycle, minstret and hpm3)
        uint64_t maj_ir, maj_cy, maj_hpm[PMU_RAW_CTR_MAX];
        int settled = 1;

        if (majority_val(w_ir, wsz, &maj_ir) < wmaj) settled = 0;
        if (settled && majority_val(w_cy, wsz, &maj_cy) < wmaj) settled = 0;

        if (settled) {
            uint64_t *col = g_col;
            for (int j = 0; j < bsz && settled; j++) {
                if (!breq->ok[j]) continue;
                for (int a = 0; a < wsz; a++) col[a] = w_hpm[a][j];
                if (majority_val(col, wsz, &maj_hpm[j]) < wmaj)
                    settled = 0;
            }
        }

        if (!settled)
            continue;

        if (first_batch) {
            *out_cy = maj_cy;
            *out_ir = maj_ir;
        }
        for (int j = 0; j < bsz; j++)
            if (breq->ok[j])
                raw_row[bs + j] = maj_hpm[j];
        return;
    }

// Fallback: reached when the single full-window vote found no majority (or the
    {
        uint64_t *col = g_col;
        uint64_t maj_ir = 0, maj_cy = 0;
        int ir_ok, cy_ok;

        ir_ok = (majority_val(w_ir, wcap, &maj_ir) >= wmaj);
        cy_ok = (majority_val(w_cy, wcap, &maj_cy) >= wmaj);

        if (first_batch) {
            if (ir_ok) {
                *out_ir = maj_ir;
            } else {
                uint64_t s = 0;
                for (int a = 0; a < wcap; a++) s += w_ir[a];
                *out_ir = s / (uint64_t)wcap;
            }
            if (cy_ok) {
                *out_cy = maj_cy;
            } else {
                uint64_t s = 0;
                for (int a = 0; a < wcap; a++) s += w_cy[a];
                *out_cy = s / (uint64_t)wcap;
            }
        }

        for (int j = 0; j < bsz; j++) {
            if (!breq->ok[j]) continue;
            for (int a = 0; a < wcap; a++) col[a] = w_hpm[a][j];
            uint64_t maj_h;
            if (majority_val(col, wcap, &maj_h) >= wmaj) {
                raw_row[bs + j] = maj_h;
            } else {
                fprintf(stderr, "[settle-spec] 0x%08x event=0x%04lx: noisy, using mean\n",
                        encoding, (unsigned long)breq->event_data[j]);
                uint64_t s = 0;
                for (int a = 0; a < wcap; a++) s += col[a];
                raw_row[bs + j] = s / (uint64_t)wcap;
                if (j == 0 && out_settled) *out_settled = 0;  /* EOI had no majority */
            }
        }
    }
}


// Prints a table header for the fixed counters
static void print_header(void)
{
    printf("%-10s  %6s  %7s\n", "Encoding", "Cycles", "Instret");
    printf("%-10s  %6s  %7s\n", "----------", "------", "-------");
}

// Prints a table row for the fixed counters
static void print_row(const probe_result_t *r)
{
    printf("0x%08x  %6lu  %7lu\n",
           r->encoding, r->cycle_delta, r->instret_delta);
}


// Create header for CSV output. Fixed counter columns plus one column per raw event.
static void write_csv_header_full(FILE *f,
                                  const struct pmu_raw_scan_result *raw_scan)
{
    fprintf(f, "encoding,cycles,instret");
    for (int ev = 0; ev < raw_scan->count; ev++)
        fprintf(f, ",0x%04llx",
                (unsigned long long)raw_scan->events[ev].event_data);
    fprintf(f, "\n");
}

// Add row to CSV output. Fixed counter columns plus one column per raw event.
static void write_csv_row_full(FILE *f,
                               const probe_result_t *r,
                               const uint64_t *raw_row,
                               const struct pmu_raw_scan_result *raw_scan)
{
    fprintf(f, "0x%08x,%lu,%lu",
            r->encoding, r->cycle_delta, r->instret_delta);
    for (int ev = 0; ev < raw_scan->count; ev++)
        fprintf(f, ",%lu", raw_row[ev]);
    fprintf(f, "\n");
}


static const struct { uint32_t enc; const char *note; } g_test_set[] = {
    /* ── Known valid: RV64GC baseline ──────────────────── */
    { 0x00000013, "NOP (addi x0,x0,0)"      },
    { 0x00000013, "NOP (addi x0,x0,0)"      },
    { 0x00000013, "NOP (addi x0,x0,0)"      },
    { 0x00000013, "NOP (addi x0,x0,0)"      },
    { 0x00000013, "NOP (addi x0,x0,0)"      },

    { 0x00000033, "ADD x0,x0,x0"            },
    { 0x40000033, "SUB x0,x0,x0"            },
    { 0x02000033, "MUL x0,x0,x0"            },
    { 0x02004033, "DIV x0,x0,x0"            },
    { 0x02005033, "DIVU x0,x0,x0"           },
    { 0x02006033, "REM x0,x0,x0"            },
    { 0x02007033, "REMU x0,x0,x0"           },

    { 0x001080b3, "ADD x1,x1,x1"            },
    { 0x401080b3, "SUB x1,x1,x1"            },
    { 0x021080b3, "MUL x1,x1,x1"            },
    { 0x021090b3, "DIV x1,x1,x1"            },
    { 0x0210a0b3, "DIVU x1,x1,x1"           },
    { 0x0210b0b3, "REM x1,x1,x1"            },
    { 0x0210c0b3, "REMU x1,x1,x1"           },

    { 0x00210133, "ADD x2,x2,x2"            },
    { 0x40210133, "SUB x2,x2,x2"            },
    { 0x02210133, "MUL x2,x2,x2"            },
    { 0x02211133, "DIV x2,x2,x2"            },
    { 0x02215133, "DIVU x2,x2,x2"           },
    { 0x02216133, "REM x2,x2,x2"            },
    { 0x02217133, "REMU x2,x2,x2"           },

    { 0x00001073, "CSRRW x0,fflags,x0"      },
    { 0x00002073, "CSRRS x0,fflags,x0"      },
    { 0xc0002073, "CSRRS x0,cycle,x0"       },

    /* ── Known valid: memory ops (may SIGSEGV on null ptr) */
    { 0x00000003, "LB x0,0(x0)"             },
    { 0x00001003, "LH x0,0(x0)"             },
    { 0x00002003, "LW x0,0(x0)"             },
    { 0x00003003, "LD x0,0(x0)"             },

    { 0x00000023, "SB x0,0(x0)"             },
    { 0x00001023, "SH x0,0(x0)"             },
    { 0x00002023, "SW x0,0(x0)"             },
    { 0x00003023, "SD x0,0(x0)"             },

    /* ── Known valid: Zba/Zbb extensions ───────────────── */
    { 0x08000033, "SHADD x0,x0,x0 (Zba)"   },
    { 0x60001013, "CTZ x0,x0 (Zbb)"        },
    { 0x60000013, "CLZ x0,x0 (Zbb)"        },

    /* ── Known invalid: reserved/nonsense patterns ─────── */
    { 0x00000000, "all-zeros (illegal)"     },
    { 0xffffffff, "all-ones  (illegal)"     },
    { 0xaaaaaaaa, "0xAAAA... (illegal)"     },
    { 0x55555555, "0x5555... (illegal)"     },

    /* ── Custom opcode space: custom-0 (0x0B) ──────────── */
    { 0x0000000b, "custom-0 funct3=0 rs1=0 rd=0" },
    { 0x0000100b, "custom-0 funct3=0 rs1=0 rd=2" },
    { 0x0000200b, "custom-0 funct3=0 rs1=0 rd=4" },
    { 0x0001000b, "custom-0 funct3=1"       },
    { 0x0002000b, "custom-0 funct3=2"       },
    { 0x0003000b, "custom-0 funct3=3"       },
    { 0x0004000b, "custom-0 funct3=4"       },
    { 0x0005000b, "custom-0 funct3=5"       },
    { 0x0006000b, "custom-0 funct3=6"       },
    { 0x0007000b, "custom-0 funct3=7"       },

    /* ── Custom opcode space: custom-1 (0x2B) ──────────── */
    { 0x0000002b, "custom-1 funct3=0"       },
    { 0x0001002b, "custom-1 funct3=1"       },
    { 0x0002002b, "custom-1 funct3=2"       },
    { 0x0003002b, "custom-1 funct3=3"       },
    { 0x0004002b, "custom-1 funct3=4"       },
    { 0x0005002b, "custom-1 funct3=5"       },
    { 0x0006002b, "custom-1 funct3=6"       },
    { 0x0007002b, "custom-1 funct3=7"       },

    /* ── Custom opcode space: custom-2 (0x5B) ──────────── */
    { 0x0000005b, "custom-2 funct3=0"       },
    { 0x0001005b, "custom-2 funct3=1"       },
    { 0x0002005b, "custom-2 funct3=2"       },
    { 0x0007005b, "custom-2 funct3=7"       },

    /* ── Custom opcode space: custom-3 (0x7B) ──────────── */
    { 0x0000007b, "custom-3 funct3=0"       },
    { 0x0001007b, "custom-3 funct3=1"       },
    { 0x0002007b, "custom-3 funct3=2"       },
    { 0x0007007b, "custom-3 funct3=7"       },

    /* ── Reserved opcodes ───────────────────────────────── */
    { 0x0000004f, "reserved opcode 0x4F"    },
    { 0x0000006f, "JAL x0,0 (valid)"        },
    { 0x0000000f, "Fence"    },
    { 0x0000008f, "Fence"    },
    { 0x000000af, "reserved opcode 0xAF"    },
    { 0x000000cf, "reserved opcode 0xCF"    },
    { 0x000000ef, "JAL x1,0 (valid)"        },
    { 0x12345678, "custom opcode (invalid)" },
    { 0x00000013, "NOP (addi x0,x0,0)"      },
    { 0x10028027, "vse128.v GhostWrite"     },
    { 0x10000027, "vse128.v GhostWrite with x0"},

    /* ── AUIPC variants ─────────────────────────────────── */
    { 0x00000017, "AUIPC x0" },
    { 0xfe007017, "AUIPC x0" },
    { 0x00000117, "AUIPC x1" },
    { 0xfe007117, "AUIPC x1" },
};

#define N_TEST (int)(sizeof(g_test_set) / sizeof(g_test_set[0]))


// Perform the full scan with all events for a single instruction encoding. Writes to CSV and stdout
static void scan_one_encoding(int fd, int cpu,
                              const struct pmu_raw_scan_result *raw_scan,
                              uint32_t enc, FILE *csv)
{
    int      first_batch = 1;
    uint64_t cy = 0, ir = 0;
    uint64_t raw_row[PMU_RAW_MAX_EVENTS];
    memset(raw_row, 0, sizeof(raw_row));

    for (int bs = 0; bs < raw_scan->count; bs++) {
        struct pmu_open_raw_request breq;
        memset(&breq, 0, sizeof(breq));
        breq.cpu   = cpu;
// Single counter per batch: the event-under-test on hpm3.
        breq.count = 1;
        breq.event_data[0] = raw_scan->events[bs].event_data;

// Flush once per batch (before opening), then settle without per-iteration flushes
        flush_slow_target();
        if (ioctl(fd, IOCTL_PMU_OPEN_RAW, &breq) == 0) {
// The single counter must open, otherwise the measurement is invalid. Warn once per run.
            if (!breq.ok[0]) {
                static int warned_partial;
                if (!warned_partial) {
                    fprintf(stderr,
                        "  WARNING: batch bs=%d failed to open hpm3 "
                        "(event 0x%04llx) - measurement invalid\n",
                        bs, (unsigned long long)breq.event_data[0]);
                    warned_partial = 1;
                }
            }
            probe_spec_batch_no_flush(enc, &breq, breq.count, bs,
                                      first_batch,
                                      &cy, &ir,
                                      raw_row, /* out_settled= */NULL);
            first_batch = 0;
        }
        ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);
    }

    /* Build a probe_result_t for print_row / CSV (fixed cols only). */
    probe_result_t r = {
        .encoding      = enc,
        .cycle_delta   = cy,
        .instret_delta = ir,
    };
    print_row(&r);
    write_csv_row_full(csv, &r, raw_row, raw_scan);
    fflush(csv);
}

// Verify mode: run a single encoding through a sliding-window stability test, writing to CSV. Returns 0 on success, nonzero on failure.
static int run_verify_mode(int fd, struct pmu_raw_scan_result *raw_scan,
                           uint32_t encoding, long n_iter,
                           int cpu, const char *csv_path);


// Full-sweep mode: run every 32-bit encoding through scan_one_encoding, writing to CSV. Returns 0 on success, nonzero on failure.
static int run_full_mode(int fd, struct pmu_raw_scan_result *raw_scan,
                         int cpu, const char *csv_path);


// Main entry point: parse args, set up PMU, and run the appropriate scan mode.
int main(int argc, char *argv[])
{
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        perror("mlockall");   /* warn, don't abort: needs CAP_IPC_LOCK / root */

    /* Optional --window N: settle-window size (default SETTLE_WINDOW_DEFAULT). */
    int settle_window = SETTLE_WINDOW_DEFAULT;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--window") != 0)
            continue;
        if (i + 1 >= argc) {
            fprintf(stderr, "  --window requires a size argument\n");
            return 1;
        }
        settle_window = (int)strtol(argv[i + 1], NULL, 0);
        for (int k = i; k + 2 < argc; k++)   /* drop argv[i] and argv[i+1] */
            argv[k] = argv[k + 2];
        argc -= 2;
        i--;                                 /* re-examine the shifted slot */
    }

    /* Optional --windows W1,W2,...: verify-mode post-hoc analysis window */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--windows") != 0)
            continue;
        if (i + 1 >= argc) {
            fprintf(stderr, "  --windows requires a comma-separated list, e.g. --windows 10,25,100\n");
            return 1;
        }
        char *list = strdup(argv[i + 1]);
        if (!list) { perror("  strdup"); return 1; }
        char *tok = strtok(list, ",");
        while (tok && g_n_analysis_windows < MAX_ANALYSIS_WINDOWS) {
            g_analysis_windows[g_n_analysis_windows++] = (int)strtol(tok, NULL, 0);
            tok = strtok(NULL, ",");
        }
        if (tok)   /* strtok still had more than MAX_ANALYSIS_WINDOWS entries */
            fprintf(stderr, "  WARNING: --windows: only first %d entries kept\n",
                    MAX_ANALYSIS_WINDOWS);
        free(list);
        for (int k = i; k + 2 < argc; k++)   /* drop argv[i] and argv[i+1] */
            argv[k] = argv[k + 2];
        argc -= 2;
        i--;
    }

    /* ── Mode / argument parsing ── */
    int         verify_mode = 0;
    int         full_mode   = 0;
    uint32_t    verify_enc  = 0;
    long        verify_iter = 0;
    const char *csv_path     = "scan_results.csv";

    /* Mode triggers: */
    const char *prog = strrchr(argv[0], '/');
    prog = prog ? prog + 1 : argv[0];
    int argbase = 1;                              /* index of <encoding_hex> */
    if (argc > 1 && strcmp(argv[1], "--verify") == 0) {
        verify_mode = 1;
        argbase = 2;
    } else if (argc > 1 && strcmp(argv[1], "--full") == 0) {
        full_mode = 1;
    } else if (strstr(prog, "verify") != NULL) {
        verify_mode = 1;
        argbase = 1;
    }

    if (verify_mode) {
        csv_path = "verify_results.csv";
        if (argc < argbase + 2) {
            fprintf(stderr,
                "usage: %s [--verify] <encoding_hex> <iterations> [csv_path]\n"
                "  e.g.: sudo %s --verify 0x00000013 40000\n"
                "  iterations: raw probe_spec_single calls per event\n",
                argv[0], argv[0]);
            return 1;
        }
        verify_enc  = (uint32_t)strtoul(argv[argbase],     NULL, 0);
        verify_iter =           strtol (argv[argbase + 1], NULL, 0);
        if (argc > argbase + 2) csv_path = argv[argbase + 2];
    } else if (full_mode) {
        csv_path = (argc > 2) ? argv[2] : "full_scan_results.csv";
    } else {
        csv_path = (argc > 1) ? argv[1] : "scan_results.csv";
    }

    int cpu = CPU_NUMBER;
    struct timeval scan_start, scan_end;

    printf("RISC-V Opcode Scanner - EIC7700X (ESWIN 770x)\n");
    printf("==============================================\n");
    printf("Mode: %s (RSB misprediction gadget)\n\n",
           verify_mode ? "VERIFY - sliding-window stability"
           : full_mode ? "FULL - exhaustive opcode sweep"
                       : "SPECULATIVE scan");

    /* ── Settle-window buffers (runtime size, heap-backed) ── */
    if (init_settle_window(settle_window) != 0)
        return 1;
    printf("  Settle window: %d (majority %d)\n",
           g_settle_window, g_settle_majority);

    if (g_n_analysis_windows == 0)   /* --windows not given: match old single-window behavior */
        g_analysis_windows[g_n_analysis_windows++] = g_settle_window;

    /* ── Pin to isolated CPU ── */
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
    printf("  Pinned to CPU %d\n", cpu);

    /* ── Open PMU device ── */
    int fd = open("/dev/pmu_monitor", O_RDWR);
    if (fd < 0) {
        perror("  open /dev/pmu_monitor");
        puts("  → sudo insmod pmu_module.ko");
        return 1;
    }

    /* ── PMU setup (cycle + instret only; HPM events via raw scan) ── */
    struct pmu_setup_info setup = { .cpu = cpu };
    if (ioctl(fd, IOCTL_PMU_SETUP, &setup) < 0) {
        perror("  IOCTL_PMU_SETUP");
        close(fd);
        return 1;
    }
    printf("  PMU setup: cycle=%s instret=%s\n",
           setup.cycle_ok   ? "ok" : "FAIL",
           setup.instret_ok ? "ok" : "FAIL");
    if (!setup.counters_running)
        puts("  WARNING: counters may be frozen - results unreliable");

    /* ── Raw event scan ── */
    static struct pmu_raw_scan_result raw_scan;
    memset(&raw_scan, 0, sizeof(raw_scan));
    if (ioctl(fd, IOCTL_PMU_RAW_SCAN, &raw_scan) < 0) {
        perror("  IOCTL_PMU_RAW_SCAN");
        /* Non-fatal: proceed with 0 raw events (CSV has no HPM columns). */
    }
    printf("  Raw scan: %d valid HPM event encodings discovered\n", raw_scan.count);
    /* Speculative scan: one event-under-test per batch, always on hpm3 (request */
    int n_batches = raw_scan.count;
    printf("  %d events, one per batch on hpm3\n\n", raw_scan.count);

    /* ── Eviction buffer ── */
    init_evict_buf();
    printf("  Eviction buffer ready\n");

    /* ── Eviction arena (2MB huge page; slow target @ offset 0) ── */
    init_evict_arena();
    printf("  Eviction arena ready (huge page, %d-line set @ stride %d)\n",
           L1D_WAYS, EVICT_STRIDE);

    /* ── Exec page ── */
    init_exec_page();
    printf("  Exec page ready\n\n");

    /* ── Probe line for the window-quality diagnostic (needs the arena) ── */
    init_probe_line();
    printf("  Probe line ready (arena+%d, set %d)\n\n",
           PROBE_LINE_OFFSET, (PROBE_LINE_OFFSET >> 6) & (L1D_SETS - 1));

    /* ── Deep-flush buffer for the window-quality diagnostic ── */
    init_deep_flush();

    /* ── Eviction quality diagnostic (non-blocking, informational) ── */
    test_eviction_quality(fd, cpu);

    /* ── Window quality diagnostic (non-blocking, informational) ── */
    test_window_quality(fd, cpu);

    /* ── Verify mode: run stability validation instead of the scan ── */
    if (verify_mode) {
        int rc = run_verify_mode(fd, &raw_scan, verify_enc, verify_iter,
                                 cpu, csv_path);
        ioctl(fd, IOCTL_PMU_TEARDOWN, NULL);
        close(fd);
        destroy_exec_page();
        return rc;
    }

    /* ── Full mode: exhaustive opcode sweep, streamed to CSV ── */
    if (full_mode) {
        int rc = run_full_mode(fd, &raw_scan, cpu, csv_path);
        ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);
        ioctl(fd, IOCTL_PMU_TEARDOWN, NULL);
        close(fd);
        destroy_exec_page();
        return rc;
    }

    /* ── CSV output ── */
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror("  fopen csv");
        close(fd);
        destroy_exec_page();
        return 1;
    }

    gettimeofday(&scan_start, NULL);

    /* ── CSV header + baseline row ── */
    write_csv_header_full(csv, &raw_scan);

    /* ── Main scan loop (speculative) ──────────────────────────────────── */
    printf("  Scanning %d instructions × %d batches × %d window size "
           "(speculative)...\n\n",
           N_TEST, n_batches ? n_batches : 1, g_settle_window);
    print_header();

    /* Per-encoding measurement + CSV streaming lives in scan_one_encoding, */
    for (int i = 0; i < N_TEST; i++)
        scan_one_encoding(fd, cpu, &raw_scan, g_test_set[i].enc, csv);

    printf("\n  Probed:     %d instructions\n", N_TEST);

    gettimeofday(&scan_end, NULL);
    double scan_seconds = (double)(scan_end.tv_sec - scan_start.tv_sec)
                        + (double)(scan_end.tv_usec - scan_start.tv_usec) / 1000000.0;
    printf("\n  Scan time:   %.3f s for %d instructions\n", scan_seconds, N_TEST);

    printf("\n  Results written to %s\n", csv_path);

    /* ── Teardown ── */
    ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);
    ioctl(fd, IOCTL_PMU_TEARDOWN, NULL);
    fclose(csv);
    close(fd);
    destroy_exec_page();
    return 0;
}

// Per-(event, analysis window) result of the raw-sample tally-based
typedef struct {
    uint64_t event_data;
    int      window;         /* analysis window size this result is for */

    uint64_t ref_majority;   /* fixed reference: mode of the first `window` recorded samples */
    int      ref_set;        /* 0 if there weren't even `window` samples recorded to analyze */
    int      ref_maj_count;  /* mode's count within the reference window, for diagnostics */

    int      tally;          /* confidence counter vs ref_majority; clamped to [0, window] */

    long     faults;         /* # post-reference samples where tally <= window/2 */
    long     total;          /* # post-reference samples processed */
} verify_result_t;

static void verify_result_init(verify_result_t *e, uint64_t event_data, int window)
{
    memset(e, 0, sizeof(*e));
    e->event_data = event_data;
    e->window     = window;
}

// Feed one recorded raw sample into an event/window's tally. Only valid
static int event_feed(verify_result_t *e, uint64_t sample)
{
    e->total++;

    if (sample == e->ref_majority) {
        if (e->tally < e->window)
            e->tally++;
    } else {
        if (e->tally > 0)
            e->tally--;
    }

    if (e->tally <= e->window / 2) {
        e->faults++;
        return 1;
    }
    return 0;
}

// Diagnostic: chunk a recorded sample timeline chronologically and print a
#define VERIFY_HIST_FAULT_THRESHOLD 0.20
#define VERIFY_HIST_CHUNKS          20
static void print_chunked_timeline_histogram(const char *what,
                                             const uint64_t *buf, long n)
{
    if (n <= 0) return;

    printf("  %s: timeline histogram (%ld samples, %d chunks):\n",
           what, n, VERIFY_HIST_CHUNKS);

    long chunk_len = n / VERIFY_HIST_CHUNKS;
    if (chunk_len < 1) chunk_len = n;   /* too few samples: one chunk */

    for (long c = 0; c * chunk_len < n; c++) {
        long start = c * chunk_len;
        long len   = chunk_len;
        if (start + len > n || c == VERIFY_HIST_CHUNKS - 1)
            len = n - start;   /* fold remainder into last chunk */
        if (len <= 0) break;

        char label[64];
        snprintf(label, sizeof label, "  chunk %2ld  samples [%ld..%ld)",
                 c, start, start + len);
        print_histogram(label, &buf[start], (int)len);

        if (start + len >= n) break;
    }
}

// Run the verify mode: for a single encoding, feed n_iter raw probe_spec_single
static int run_verify_mode(int fd, struct pmu_raw_scan_result *raw_scan,
                           uint32_t encoding, long n_iter,
                           int cpu, const char *csv_path)
{
// One batch per event, each batch measures that event on hpm3.
    int n_batches = raw_scan->count;

    printf("Raw-sample tally stability validator (measure-then-analyze)\n");
    printf("=============================================================\n");
    printf("  encoding:          0x%08x\n", encoding);
    printf("  raw samples:       %ld per event (plus %d cold-start discard)\n",
           n_iter, g_settle_window);
    printf("  analysis windows: ");
    for (int i = 0; i < g_n_analysis_windows; i++)
        printf(" %d", g_analysis_windows[i]);
    printf("\n");
    printf("  %d events, one per batch on hpm3\n\n", raw_scan->count);

// Write candidate to RSB gadget
    g_exec_page[GADGET_CANDIDATE] = encoding;
    __builtin___clear_cache((char *)&g_exec_page[GADGET_CANDIDATE],
                            (char *)(&g_exec_page[GADGET_CANDIDATE] + 1));

    FILE *csv = fopen(csv_path, "w");
    if (csv)
        fprintf(csv, "event,window,faults,total,fault_rate_pct,ref_majority,ref_maj_count,tally_final\n");

// Full post-discard timeline buffer, reused per event: every sample
    uint64_t *hist_buf = malloc((size_t)n_iter * sizeof *hist_buf);
    if (!hist_buf) { perror("  malloc hist_buf"); if (csv) fclose(csv); return 1; }

// Parallel mcycle-delta recording, same samples as hist_buf. Costs
    uint64_t *cyc_buf = malloc((size_t)n_iter * sizeof *cyc_buf);
    if (!cyc_buf) { perror("  malloc cyc_buf"); free(hist_buf); if (csv) fclose(csv); return 1; }

// Per (event, analysis window) result.
    verify_result_t *results =
        calloc((size_t)raw_scan->count * g_n_analysis_windows, sizeof *results);
    if (!results) {
        perror("  calloc results");
        free(hist_buf); free(cyc_buf); if (csv) fclose(csv);
        return 1;
    }

    for (int bs = 0; bs < raw_scan->count; bs++) {
        struct pmu_open_raw_request breq;
        memset(&breq, 0, sizeof(breq));
        breq.cpu   = cpu;
// Single counter per batch: the event-under-test on hpm3.
        breq.count = 1;
        breq.event_data[0] = raw_scan->events[bs].event_data;

        flush_slow_target();
        if (ioctl(fd, IOCTL_PMU_OPEN_RAW, &breq) < 0) {
            fprintf(stderr, "  OPEN_RAW failed for event %d, skipping\n", bs);
            continue;
        }

// The single counter must open, otherwise the measurement is invalid.
        if (!breq.ok[0]) {
            static int warned_partial;
            if (!warned_partial) {
                fprintf(stderr,
                    "  WARNING: event %d failed to open hpm3 "
                    "(event 0x%04llx) - measurement invalid\n",
                    bs, (unsigned long long)breq.event_data[0]);
                warned_partial = 1;
            }
        }

// ── Phase A: measurement. Every iteration does exactly the same
        long hist_n = 0;
        for (long iter = 0; iter < g_settle_window + n_iter; iter++) {
            flush_slow_target();
            uint64_t touch = g_slow_target; (void)touch;
            evict_slow_target_l2();
            raw_sample_t s = probe_spec_single(&breq, breq.count);

            if (!breq.ok[0])
                continue;               /* counter never opened - nothing to record */
            if (iter < g_settle_window)
                continue;               /* cold-start discard: warm the pipeline only */

            hist_buf[hist_n]   = s.hpm[0];
            cyc_buf[hist_n]    = s.cycle;
            hist_n++;
        }

        ioctl(fd, IOCTL_PMU_CLOSE_RAW, NULL);

// ── Phase B: analysis. No more probes are issued for this event -
        int any_high_fault = 0;
        for (int wi = 0; wi < g_n_analysis_windows; wi++) {
            int W = g_analysis_windows[wi];
            verify_result_t *e = &results[(size_t)bs * g_n_analysis_windows + wi];
            verify_result_init(e, raw_scan->events[bs].event_data, W);

            if (hist_n < W) {
                fprintf(stderr,
                    "  WARNING: event 0x%04llx window=%d: only %ld samples "
                    "recorded, need >= %d - skipping\n",
                    (unsigned long long)e->event_data, W, hist_n, W);
                continue;
            }

            int cnt = majority_val(hist_buf, W, &e->ref_majority);
            e->ref_maj_count = cnt;
            e->tally         = cnt > W ? W : cnt;
            e->ref_set       = 1;
            if (cnt < W / 2 + 1)
                fprintf(stderr,
                    "  WARNING: event 0x%04llx window=%d: no strict majority "
                    "in reference window (best=%d/%d) - using most common "
                    "value as reference\n",
                    (unsigned long long)e->event_data, W, cnt, W);

            for (long i = W; i < hist_n; i++)
                event_feed(e, hist_buf[i]);

            double rate = e->total > 0
                ? 100.0 * (double)e->faults / (double)e->total : 0.0;
            printf("Event %d/%d  0x%04llx  window=%-5d: faults=%ld / %ld  (%.4f%%)  "
                   "ref_maj=%lu (%d/%d)  tally_final=%d\n",
                   bs + 1, n_batches,
                   (unsigned long long)e->event_data, W,
                   e->faults, e->total, rate,
                   (unsigned long)e->ref_majority, e->ref_maj_count, W,
                   e->tally);
            if (csv) {
                fprintf(csv, "0x%04llx,%d,%ld,%ld,%.4f,%lu,%d,%d\n",
                        (unsigned long long)e->event_data, W,
                        e->faults, e->total, rate,
                        (unsigned long)e->ref_majority, e->ref_maj_count, e->tally);
                fflush(csv);
            }

            if (e->total > 0 && rate >= VERIFY_HIST_FAULT_THRESHOLD * 100.0)
                any_high_fault = 1;
        }

        if (any_high_fault && hist_n > 0) {
            char label[64];
            snprintf(label, sizeof label, "0x%04llx (hpm3)",
                     (unsigned long long)raw_scan->events[bs].event_data);
            print_chunked_timeline_histogram(label, hist_buf, hist_n);

            snprintf(label, sizeof label, "0x%04llx (mcycle, same samples)",
                     (unsigned long long)raw_scan->events[bs].event_data);
            print_chunked_timeline_histogram(label, cyc_buf, hist_n);
        }
    }

    if (csv) fclose(csv);
    free(results);
    free(hist_buf);
    free(cyc_buf);
    printf("Results written to %s\n", csv_path);
    return 0;
}


// Full mode: run every 32-bit encoding through scan_one_encoding, writing to CSV. Returns 0 on success, nonzero on failure.
static int run_full_mode(int fd, struct pmu_raw_scan_result *raw_scan,
                         int cpu, const char *csv_path)
{
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror("  fopen csv");
        return 1;
    }
    write_csv_header_full(csv, raw_scan);

    const long n_total = 28L * 8L * 128L;   /* 28,672 encodings */
    long       n_done  = 0;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    printf("  Full sweep: %ld encodings (28 opcodes x 8 funct3 x 128 funct7)\n",
           n_total);
    printf("  rd=rs1=rs2=x0; one event/batch on hpm3; streaming to %s\n\n",
           csv_path);
    print_header();

    for (uint32_t opcode = 0x03; opcode < 0x80; opcode += 4) {
        if (((opcode >> 2) & 0x7) == 0x7)
            continue;   /* skip >=48-bit length markers (bits[4:2]=111) */

// Print progress to stderr per opcode (28x)
        gettimeofday(&t1, NULL);
        double elapsed = (double)(t1.tv_sec - t0.tv_sec)
                       + (double)(t1.tv_usec - t0.tv_usec) / 1e6;
        fprintf(stderr,
                "[full] opcode 0x%02x  %ld/%ld encodings (%.1f%%)  elapsed %.1fs\n",
                opcode, n_done, n_total,
                100.0 * (double)n_done / (double)n_total, elapsed);

        for (uint32_t funct3 = 0; funct3 < 8; funct3++)
            for (uint32_t funct7 = 0; funct7 < 128; funct7++) {
                uint32_t enc = (funct7 << 25) | (funct3 << 12) | opcode;
                scan_one_encoding(fd, cpu, raw_scan, enc, csv);
                n_done++;
            }
    }

    gettimeofday(&t1, NULL);
    double total_s = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_usec - t0.tv_usec) / 1e6;

    fclose(csv);
    printf("\n  Full sweep complete: %ld encodings in %.1fs\n", n_done, total_s);
    printf("  Results written to %s\n", csv_path);
    return 0;
}
