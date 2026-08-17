# riscv-pmu-scanner

Scanning RISC-V for undocumented instructions using transient execution and
performance-counter signatures, with no dependence on the illegal-instruction
fault.

This is the artifact for the MSc thesis *Scanning for undocumented and
anomalous RISC-V instruction encodings* (VUSec). It scans the instruction
encoding space of a RISC-V core, decides which encodings the processor
actually acts on, and characterises the ones its own toolchain cannot name —
including the four CVE-2024-44067 (GhostWrite) encodings, which the method
isolates without being told what it is looking for.

## What it does

A conventional instruction scanner learns whether an encoding is valid by
executing it and watching for an illegal-instruction fault. That signal is
lossy: the processor can do substantial work for an encoding and *still* refuse
it, and a fault-based scanner records only the refusal.

This scanner replaces the fault with a different oracle. Each candidate
encoding is executed **transiently**, behind a return-stack misprediction, so
it runs speculatively and is squashed before it commits or faults. What it did
on the way to being squashed is read from the hardware performance counters as
a per-encoding **signature**. The candidate never retires — `instret` reads
zero for all 28,672 encodings scanned — so every counter that moves is
reporting work the processor performed on an encoding it never architecturally
executed.

Three stages turn those signatures into results:

1. **Baseline** — an encoding is *active* if any counter deviates from the
   harness baseline, *inert* otherwise. Threshold-free.
2. **Ground truth** — an external decoder (GNU binutils) labels each encoding
   *documented* or *undocumented*. Crossing this with activity gives a 2×2
   matrix; the interesting cell is undocumented-yet-active.
3. **Classification** — for each undocumented-active encoding: an exact-twin
   lookup, a supervised operation classifier, and a model-free anomaly triage
   that ranks encodings by how far they sit outside their opcode's documented
   behaviour.

## Key findings

On the Milk-V Pioneer (SG2042, T-Head XuanTie C920):

- Of 28,672 encodings scanned, **11,225 are active** and **635 are active but
  undocumented** — the processor does work the toolchain cannot name.
- **302 of those 635 raise `SIGILL`** when executed for real, yet move
  performance counters anyway (278 on three or more counters). A scanner whose
  oracle is the fault signal reports all 302 as nonexistent. This is the
  central result: **refusal and activity are separate facts.**
- A **model-free triage**, given no vulnerability knowledge and no tuned
  threshold, places the four GhostWrite encodings (`0x10000027` and three
  siblings) alone in a 4-of-635 bucket. On the same counters a trained
  classifier labels all four `load`, at a *store* opcode — the failure of the
  learned label is itself part of the finding.

The scan **isolates** a known-vulnerable encoding blind; it does not claim to
discover GhostWrite, which was already public. Recovering it without bug
knowledge is the evidence that signature novelty is a usable bug-hunting
signal.

See the thesis for the full treatment.

## Repository layout

```
.                    board-side C + Makefile live in the root
  pmu_module.c           delegates the PMU to userspace via /dev/pmu_monitor
  pmu_scanner.c          the speculative scanner (batch / verify / full modes)
  pmu_test.c             counter-availability smoke test
  arch_fault_scanner.c   executes encodings for real, records the fault
  Makefile               builds all of the above + the module
  test.sh                rebuild, reload, and verify the module
analysis/            host-side Python: runs on the scan CSVs
  analyze_runs.py            5 runs -> consensus_scan.csv, reliability figures
  binutils_screen.py         the documented/undocumented decoder screen
  screen_report.py           confusion matrix + architectural-fault tables
  classify_instructions.py   supervised functional classifier
  signature_classify.py      operation calls for undocumented-active encodings
  triage.py                  model-free anomaly triage (imported, not run)
  figs_anomaly.py            the anomaly-triage figures
  opcode_activity_plot.py    per-opcode activity figure
  kernel.py                  shared matplotlib styling
  requirements.txt
outputs/             the data produced for the thesis, committed so the
                     analysis can be reproduced without the board
  scan_run{1..5}.csv         the five raw sweeps
  consensus_scan.csv         their consensus (input to every analysis step)
  binutils_screen_arch.csv   the architectural-fault results
  verify_results.csv         the window-calibration sweep
```

## Requirements

**Board.** A Milk-V Pioneer (SG2042 / T-Head C920). The scanner's cache
geometry and measurement core are specific to this board, selected by the
`-DTARGET_MILKV_PIONEER` flag the Makefile sets from the device-tree compatible
strings.

The **kernel version is not free to choose.** Newer kernels disable the T-Head
vector extension that CVE-2024-44067 belongs to, which would leave the
GhostWrite encodings inactive and the headline result unreproducible. The
evaluated system is:

- Linux **6.1.31** (riscv64), OpenSBI 1.2
- GCC **13.1.1** (compiled on the board)
- kernel headers / build tree under `/lib/modules/$(uname -r)/build`

**Analysis host.** Any machine.

- Python **3.9.6** (the pinned package versions are the last supporting 3.9)
- GNU binutils with XTheadVector support (evaluated: 2.47) — must decode
  `th.vsb.v` under `-march=rv64gc_xtheadvector`

## Running

The pipeline is strictly ordered; each step consumes the previous one's
output, and the scan data crosses between the board and the analysis host
twice. The commands below write regenerated artifacts to `analysis/out/` (the
scripts' default `--outdir`), leaving the committed `outputs/` untouched.

### 1. Build and smoke-test (board)

```bash
make                # pmu_test, pmu_scanner, arch_fault_scanner, pmu_module.ko
./test.sh           # rebuild, load the module, run pmu_test
```

`pmu_test` reports how many hardware counters the platform grants and whether
`cycle`/`instret` are readable from userspace. If no counters open, the
delegation is wrong and every later measurement reads zero.

Isolate the measurement core (CPU 63 on the Pioneer) before scanning, or
cross-run agreement degrades. Reserve its NUMA node at boot:

```
isolcpus=40-47,56-63 nohz_full=40-47,56-63 rcu_nocbs=40-47,56-63 irqaffinity=0-39
```

The scanner needs 65 × 2 MB huge pages free **on that node**.

### 2. Scan (board)

Five independent full sweeps at window 256. One sweep takes a little under two
hours on the Pioneer.

```bash
for i in 1 2 3 4 5; do
  sudo taskset -c 63 chrt -f 99 ./pmu_scanner --full --window 256 scan_run$i.csv
done
```

Then unload the module — it grants userspace direct PMU access and should not
be left loaded:

```bash
make unload
```

The five sweeps used for the thesis are in `outputs/`, so this step can be
skipped to reproduce only the analysis.

### 3. Consensus + reliability (host)

```bash
cd analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 analyze_runs.py ../outputs/scan_run{1,2,3,4,5}.csv --outdir out
```

Produces `analysis/out/consensus_scan.csv` — the artifact every later step
consumes — and the cross-run reliability figures. (The committed
`outputs/consensus_scan.csv` is the same file; point the later steps at
whichever you prefer.)

### 4. Decoder screen (host)

```bash
python3 binutils_screen.py --scan out/consensus_scan.csv --outdir out
```

Produces `out/binutils_screen.csv`, with 635 rows marked
`undocumented_active`.

### 5. Architectural cross-reference (board, then host)

Copy `out/binutils_screen.csv` to the board:

```bash
./arch_fault_scanner --screen binutils_screen.csv \
                     --out binutils_screen_arch.csv
```

> This executes arbitrary undocumented instructions in userspace. On the
> evaluated hardware they terminate cleanly, but run it only on a machine you
> are willing to reset.

Copy `binutils_screen_arch.csv` back beside `binutils_screen.csv` on the host
(the committed copy is in `outputs/`), then:

```bash
python3 screen_report.py --scan out/consensus_scan.csv --outdir out
```

`screen_report.py` picks up the `_arch` sidecar automatically. Produces the
confusion matrix and `out/undocumented_active_arch_faults.csv`: **302 sigill,
68 sigsegv, 265 none**.

### 6. Classification and triage (host)

```bash
python3 classify_instructions.py out/consensus_scan.csv --outdir out
python3 opcode_activity_plot.py  --scan out/consensus_scan.csv --outdir out
python3 signature_classify.py    --scan out/consensus_scan.csv \
                                 --screen out/binutils_screen.csv --outdir out
python3 figs_anomaly.py          --scan out/consensus_scan.csv \
                                 --screen out/binutils_screen.csv --outdir out
```

`figs_anomaly.py` writes `fig_triage_novelty.png`, in which the four GhostWrite
encodings are the only tier-B points. Tier populations: A 144, B 4, C 68,
D 419.

## Reproducing the headline result without the board

Steps 3–6 run against the committed `outputs/consensus_scan.csv` and
`outputs/binutils_screen_arch.csv`, and need no hardware. This reproduces the
confusion matrix, the architectural-fault breakdown, the classification, and
the triage — everything except the raw scan itself, which requires the
Pioneer.

## Window calibration (optional)

`outputs/verify_results.csv` is the sweep behind the settle-window choice. To
regenerate it on the board:

```bash
sudo taskset -c 63 chrt -f 99 ./pmu_scanner --verify 0x00000013 1000000 \
     --windows 8,16,32,64,128,256,512,1024,2048 verify_results.csv
```

Fault rate falls to zero for all retained events by window 128; the scan uses
256, a doubling above that. Events `0x0027` and `0x0028` never converge and are
dropped.

## License

MIT — see [`LICENSE`](LICENSE). One exception: `pmu_module.c` is a Linux kernel
module and is additionally licensed GPL-2.0, which the kernel requires of any
loadable module. That file may be used under either license.

## Citation

If you use this work, please cite the thesis. A BibTeX entry will be added on
publication.