# PAQ — Per-page Adaptive Q-learning for Page Replacement

A page replacement policy that learns **what an evictable page looks like**, instead of
following a fixed rule or picking between existing ones.

Operating Systems course project (BCSE303L), Vellore Institute of Technology, Vellore.

**Pious Bansal** (24BCE0421) · **Urvika Lamba** (24BCE0348)

---

## The problem

When RAM is full and a program needs a page that isn't resident, the operating system has to
throw something out. A page fault costs roughly 100,000 times more than a hit, so that choice
directly affects how fast the machine runs.

Every classical policy is a fixed guess about program behaviour, and every fixed guess has a
workload that defeats it. We measured this before designing anything — on a loop of 9 pages
with 8 frames, over 20,000 references:

| Policy | Faults | Hit rate |
|---|---|---|
| FIFO | 20000 / 20000 | 0% |
| LRU | 20000 / 20000 | 0% |
| Belady (optimal, needs the future) | 2507 | 87.5% |

LRU is not merely poor here. In a loop, the least-recently-used page is always the very next
page needed, so it is wrong on every single decision.

## The idea

PAQ describes each resident page by two numbers it already has — how long since it was last
used, and how many times it has been used — and buckets each on a doubling scale into 8 bands.
That gives **64 page types**. A single table holds one eviction score per type.

On a fault, every resident page is labelled with its type and the highest-scoring type is
evicted. The score is learned from what happens next:

- The evicted page is recorded as a small **receipt** (page number, its type, the time). The
  page's contents are discarded; the receipt is about 12 bytes.
- If that page is requested again within a window, the eviction was a mistake → push that
  type's score toward −1.
- If the receipt expires untouched, the eviction was correct → push it toward +1.

So the delayed consequence of an eviction becomes a usable training signal, and learning
continues during normal operation. There is no training phase, no neural network, and no
oracle.

The reason 64 entries is enough: **pages are interchangeable given their features.** It does
not matter which page it is, only what it looks like, so a lesson learned from one page applies
immediately to every other page that looks the same — including pages never seen before.
Treating the whole contents of RAM as the state would instead need roughly 10^14 entries.

## Build and run

No libraries, no dependencies, no GPU.

```bash
g++ -std=c++17 -O2 paq.cpp -o paq
./paq
```

Runs the correctness gates, then the full result set, in a few seconds. It also runs unmodified
in a browser-based C++ compiler.

## Correctness gates

These run before any measurement. If one fails, the simulator is wrong and no number after it
is worth anything.

| # | Test | Result |
|---|---|---|
| 1 | No policy may ever beat Belady on any trace | PASS |
| 2 | FIFO must show Belady's Anomaly on the textbook string | PASS — 9 faults at 3 frames, 10 at 4 |
| 3 | LRU must score ~0% hits on a loop longer than RAM | PASS — 4000/4000 faults |
| 4 | LRU must beat FIFO on a locality-heavy trace | PASS — 3700 vs 4099 |

Gate 2 is the interesting one. Belady's Anomaly is the counter-intuitive fact that FIFO can
produce *more* faults when given *more* memory. Reproducing it is evidence the simulator models
the subsystem faithfully rather than approximately.

## Results

Fault counts at 8 frames. Lower is better. Belady requires the entire future, so it is a
ceiling to measure against, not a competitor.

| Workload | Random | FIFO | CLOCK | LRU | MRU | LFU | **PAQ** | Belady | Gap closed |
|---|---|---|---|---|---|---|---|---|---|
| Scan | 20000 | 20000 | 20000 | 20000 | 20000 | 20000 | **20000** | 20000 | 0.0% |
| Loop | 4425 | 20000 | 20000 | 20000 | 2507 | 20000 | **4240** | 2507 | **90.1%** |
| Zipf | 13672 | 13693 | 13368 | 13007 | 16460 | 9871 | **11864** | 8213 | 23.8% |
| Phase-shift | 33048 | 37042 | 38011 | 46296 | 48277 | 29714 | **31379** | 26417 | **75.0%** |

"Gap closed" is `(LRU − ours) / (LRU − Belady)` — the share of the *available* improvement
captured. Raw fault counts are not comparable across workloads whose headroom differs by
orders of magnitude.

### The advantage grows with memory

Repeated at four memory sizes. The loop and phase-shift traces are **regenerated at each size**
so the loop stays marginally larger than RAM; reusing a fixed loop would let it fit entirely in
memory at 32 frames, where LRU is already optimal and nothing is being measured.

| Workload | 4 frames | 8 frames | 16 frames | 32 frames |
|---|---|---|---|---|
| Scan | 0.0% | 0.0% | 0.0% | 0.0% |
| Loop | 79.0% | 90.1% | 95.4% | **98.1%** |
| Zipf | 19.7% | 23.8% | 27.3% | 24.0% |
| Phase-shift | 63.9% | 74.8% | 82.5% | **91.9%** |

More frames means more candidate pages to rank, so a learned ordering has more room to beat a
fixed rule.

### Learning during execution matters

PAQ learns while it runs. Freezing the table after an initial period and repeating the
phase-shift run at 8 frames isolates how much that contributes:

| Configuration | Faults | Gap closed |
|---|---|---|
| LRU | 46,296 | baseline |
| Belady (not implementable) | 26,417 | 100% by definition |
| PAQ, table frozen | 37,042 | 46.6% |
| **PAQ, learning online** | **31,379** | **75.0%** |

Continuous learning accounts for a further 5,663 faults avoided. The 75.0% figure is stable
across nine different combinations of exploration rate and learning rate, so it is not a
hyperparameter accident.

### What it learned

With only 64 entries, the learned policy can simply be printed and read. On the Zipf workload:

```
             f0     f1     f2     f3     f4     f5     f6     f7
  r0      0.04   0.55   0.85   0.84   0.69   0.87   0.83  -0.52
  r1      0.77   0.35   0.59   0.91   0.55   0.77   0.82  -0.40
  r2      0.29   0.09   0.56   0.35   0.84   0.78   0.66  -0.07
  r3      0.19   0.04   0.52   0.82   0.50   0.34   0.56  -0.03
  r4      0.52   0.11   0.45   0.31   0.04  -0.61   0.03   0.46
  r5      0.00  -0.19  -0.03   0.00  -0.37  -0.10   0.20  -0.00
```

Rows are recency bands, columns frequency bands. Column `f7` holds the most frequently
referenced pages, and every value in it is negative or near zero — the agent worked out on its
own that frequently used pages must not be evicted. Columns `f1`–`f6` are strongly positive:
rarely used pages are safe to remove.

**No such rule was supplied.** PAQ received only a reward signal and arrived at behaviour
resembling a frequency-based policy, which happens to be the right answer for a skewed
workload.

## Where it loses

Reported honestly, because a results table with no losses in it is not believable.

- **Zipf: LFU wins** (9,871 vs 11,864 faults). An ablation across nine parameter settings does
  not move this. It follows from bucketing reference counts into 8 bands, which discards the
  exact counts LFU exploits — the price of keeping the learned state small and readable.
- **Scan: nothing helps.** Every policy including Belady faults on every reference. A 0% gap
  closed here means no headroom existed, not that the method failed.
- **Loop: MRU hits the optimum exactly** (2507). Evicting the most recently used page is
  precisely correct for that pattern. PAQ reaches 90.1% of the available reduction while
  discovering the rule from observation rather than being handed it.

The shape of these results is the point: **PAQ wins where the workload changes character, and
ties or loses where a single fixed heuristic already fits.**

## What's in here

```
paq.cpp      Complete implementation — workloads, 7 baseline policies,
             Belady's optimal, the PAQ agent, correctness gates, all experiments
README.md    This file
```

Everything is one self-contained C++17 file using only the standard library. The sections
inside it are: workload generators, baseline policies, the PAQ agent (feature extractor,
Q-table, victim selector, ghost list), the gap-closed metric, the correctness gates, and the
experiment runner.

## Limitations

Evaluation is by trace-driven simulation. PAQ has not been integrated into a production
operating system kernel, and the per-decision cost of computing features across resident pages
is higher than LRU's O(1) with the right data structure. Both are measured and stated rather
than elided.

Real memory traces captured with `valgrind --tool=lackey --trace-mem=yes` are the next step;
results here are from the four seeded synthetic generators.

## Status

Patent pending. An invention disclosure has been filed through the VIT IPR & TT Cell.
All rights reserved.
