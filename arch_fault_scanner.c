#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGE_SIZE 4096U
#define CSV_MAX_FIELDS 32
#define PROBE_TIMEOUT_SEC 1

static sigjmp_buf g_env;
static volatile sig_atomic_t g_signal = 0;
static volatile sig_atomic_t g_timed_out = 0;
static volatile sig_atomic_t g_in_probe = 0;

static void die(const char *msg)
{
    perror(msg);
    exit(1);
}

static char *xstrdup(const char *s)
{
    char *out = strdup(s ? s : "");
    if (!out) {
        die("strdup");
    }
    return out;
}

static void strip_newline(char *s)
{
    if (!s) {
        return;
    }
    s[strcspn(s, "\r\n")] = '\0';
}

static size_t csv_split(char *line, char **fields, size_t max_fields)
{
    size_t n = 0;
    char *cursor = line;

    while (n < max_fields) {
        fields[n++] = strsep(&cursor, ",");
        if (cursor == NULL) {
            break;
        }
    }
    return n;
}

static int find_field(char **fields, size_t n_fields, const char *name)
{
    for (size_t i = 0; i < n_fields; ++i) {
        if (fields[i] && strcmp(fields[i], name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static void trap_handler(int sig, siginfo_t *info, void *ctx)
{
    (void)info;
    (void)ctx;
    g_signal = sig;
    g_timed_out = (sig == SIGALRM);
    if (g_in_probe) {
        siglongjmp(g_env, 1);
    }
    _exit(128 + sig);
}

static void install_handlers(void)
{
    struct sigaction sa;

    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = trap_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);

    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGTRAP, &sa, NULL);
    sigaction(SIGALRM, &sa, NULL);
}

static void *alloc_exec_page(void)
{
    void *p = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        die("mmap RWX page");
    }
    return p;
}

static const char *fault_from_signal(int sig, int timed_out)
{
    if (timed_out || sig == SIGALRM) {
        return "timeout";
    }

    switch (sig) {
    case 0:       return "none";
    case SIGILL:  return "sigill";
    case SIGSEGV: return "sigsegv";
    case SIGBUS:  return "sigbus";
    case SIGFPE:  return "sigfpe";
    case SIGTRAP: return "sigtrap";
    default:      return "signal";
    }
}

static void run_probe(uint32_t encoding, char *fault_buf, size_t fault_buf_sz)
{
    void *page = alloc_exec_page();
    uint32_t *code = (uint32_t *)page;
    typedef void (*fn_t)(void);

    /* A tiny RISC-V function body: raw instruction, then a valid return. */
    code[0] = encoding;
    code[1] = 0x00000013U; /* nop */
    code[2] = 0x00008067U; /* ret */
    __builtin___clear_cache((char *)page, (char *)page + PAGE_SIZE);

    g_signal = 0;
    g_timed_out = 0;
    g_in_probe = 1;
    alarm(PROBE_TIMEOUT_SEC);

    if (sigsetjmp(g_env, 1) == 0) {
        fn_t fn = (fn_t)page;
        fn();
        alarm(0);
        g_in_probe = 0;
        snprintf(fault_buf, fault_buf_sz, "%s", fault_from_signal(0, 0));
        munmap(page, PAGE_SIZE);
        return;
    }

    alarm(0);
    g_in_probe = 0;
    snprintf(fault_buf, fault_buf_sz, "%s",
             fault_from_signal((int)g_signal, (int)g_timed_out));
    munmap(page, PAGE_SIZE);
}

static FILE *open_screen_csv(const char *requested_path)
{
    FILE *fp = NULL;

    if (requested_path != NULL) {
        fp = fopen(requested_path, "r");
        if (!fp) {
            fprintf(stderr, "failed to open %s: %s\n", requested_path,
                    strerror(errno));
            exit(1);
        }
        return fp;
    }

    const char *candidates[] = {
        "analysis/out/binutils_screen.csv",
        "out/binutils_screen.csv",
        "binutils_screen.csv",
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        fp = fopen(candidates[i], "r");
        if (fp) {
            return fp;
        }
    }

    fprintf(stderr,
            "could not open binutils_screen.csv; tried analysis/out, out, and cwd\n");
    exit(1);
}

static FILE *open_output_csv(const char *path)
{
    if (path == NULL || strcmp(path, "-") == 0) {
        return stdout;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "failed to open %s for writing: %s\n", path,
                strerror(errno));
        exit(1);
    }
    return fp;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: %s [--screen PATH] [--out PATH]\n"
            "  --screen  binutils_screen.csv to filter (default: analysis/out/binutils_screen.csv)\n"
            "  --out     output csv path, or '-' for stdout (default: '-')\n",
            argv0);
}

int main(int argc, char **argv)
{
    const char *screen_path = NULL;
    const char *out_path = "-";
    FILE *in = NULL;
    FILE *out = NULL;
    char *line = NULL;
    size_t cap = 0;
    ssize_t nread;
    char *fields[CSV_MAX_FIELDS];
    int encoding_idx = -1;
    int verdict_idx = -1;
    size_t selected = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--screen") == 0 && i + 1 < argc) {
            screen_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    install_handlers();
    in = open_screen_csv(screen_path);
    out = open_output_csv(out_path);

    nread = getline(&line, &cap, in);
    if (nread < 0) {
        die("getline header");
    }
    strip_newline(line);

    char *header_copy = xstrdup(line);
    size_t n_header = csv_split(header_copy, fields, CSV_MAX_FIELDS);
    encoding_idx = find_field(fields, n_header, "encoding");
    verdict_idx = find_field(fields, n_header, "verdict");
    if (encoding_idx < 0 || verdict_idx < 0) {
        fprintf(stderr, "screen csv is missing encoding/verdict columns\n");
        free(header_copy);
        free(line);
        fclose(in);
        if (out != stdout) {
            fclose(out);
        }
        return 1;
    }

    fprintf(out, "%s,arch_fault\n", line);

    while ((nread = getline(&line, &cap, in)) >= 0) {
        char *row_copy;
        uint32_t encoding;
        char fault[32];
        size_t n_fields;

        strip_newline(line);
        if (line[0] == '\0') {
            continue;
        }

        row_copy = xstrdup(line);
        n_fields = csv_split(row_copy, fields, CSV_MAX_FIELDS);
        if ((size_t)encoding_idx >= n_fields || (size_t)verdict_idx >= n_fields) {
            free(row_copy);
            continue;
        }

        if (strcmp(fields[verdict_idx], "undocumented_active") != 0) {
            free(row_copy);
            continue;
        }

        encoding = (uint32_t)strtoul(fields[encoding_idx], NULL, 0);
        run_probe(encoding, fault, sizeof(fault));
        fprintf(out, "%s,%s\n", line, fault);
        fflush(out);
        ++selected;
        free(row_copy);
    }

    free(header_copy);
    free(line);
    fclose(in);
    if (out != stdout) {
        fclose(out);
    }

    fprintf(stderr, "wrote %zu undocumented_active rows with arch_fault column\n",
            selected);
    return selected > 0 ? 0 : 1;
}