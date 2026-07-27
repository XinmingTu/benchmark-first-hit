# Benchmark First Hit

**Benchmark First Hit measures when fixed score mass first enters the best
observed frontier of any model-and-agent system.**

The curated corpus contains **712 sourced observations**, **47 fixed
benchmark/version definitions**, **44 eligible frontier series**, and a locked
**31-benchmark panel**, through 2026-07-21.

This is an empirical first-hit index and a candidate theory of capability
diffusion under selective measurement. It is not yet a universal scaling law.

![Benchmark First Hit measurement](output/first_hit_measurement.png)

## The central idea

For benchmark \(b\), let \(m_b(\tau)\in[0,1]\) be its best corpus score by date
\(\tau\). If \(Q\sim\mathrm{Uniform}(0,1)\) and

\[
T_{bQ}=\inf\{\tau:m_b(\tau)\ge Q\},
\]

then exactly

\[
\boxed{
\frac1B\sum_bm_b(\tau)
=
\Pr_{b,Q}(T_{bQ}\le\tau)
}.
\]

The equal-benchmark frontier is therefore the empirical CDF of score-threshold
first-hit dates. This identity is always true; whether the CDF follows a
logistic, Gompertz, or another law is an empirical question.

## What is held fixed

Any base model, agent scaffold, tool stack, ensemble, or inference budget may
set the frontier. The benchmark edition, evaluated task set, metric,
denominator, and score meaning must remain comparable.

Every benchmark contributes one unit of native 0–100 score mass. This
normalizes range, not semantic difficulty.

## Three dates, three questions

| Clock | Meaning | Status here |
|---|---|---|
| \(U\) | first latent availability of a capable system | not identified |
| \(R^{\mathcal C}\) | earliest system date assigned by the frozen corpus | current primary axis |
| \(T^{\mathcal C}\) | first verified public disclosure | incomplete |

The present result is a **corpus-attributed retrospective frontier**. Later
back-tests can revise \(R^{\mathcal C}\) backward. A true real-time forecast
requires frozen historical corpus snapshots and disclosure dates.

## Current empirical result

- All 44 series: **45.7%** left/prevalent mass, **35.7%** later dated jump
  mass, and **18.6%** unresolved mass.
- Fixed 31 panel: **35.4% / 46.3% / 18.3%** respectively.
- The fixed panel rises **9.86 points** from 2025-08-07 to 2026-07-09.
- In a delayed-entry retrospective backtest, logistic 90/180-day RMSE is
  **2.56 / 5.07** points versus **5.37 / 10.68** for no-change over 12/11
  origins.
- Gompertz is slightly better in sample, while vintage winners vary among
  Gompertz, logistic, and probit. There is no uniquely identified link
  function.
- A 4,000-replicate stationary-score, schedule-preserving null gives
  \(p=0.00025\), but **18/31** fixed-panel benchmarks are more than one year
  stale.

The robust result is structured accumulation of first-hit mass. A fitted
ceiling and a causal learning rate remain unidentified.

## Read the analysis

- [PAPER.md](PAPER.md): estimand, empirical design, results, forecasts, nulls,
  and prospective test.
- [THEORY.md](THEORY.md): first-hit theorem, IRT threshold bridge,
  exposure–threshold model, unlock/detection process, identification limits,
  and falsifiable predictions.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
python3 scripts/analyze.py --no-plots
python3 scripts/analyze_unlocks.py
python3 scripts/plot_paper.py
```

The full default run is deterministic. Generated audit CSVs remain under
`output/`; the public repository tracks only three compact headline renders.

## Repository map

- `data/benchmark_observations.csv` — scores, system configurations, date
  semantics, source URLs, and eligibility.
- `data/benchmark_metadata.csv` — exact benchmark editions, metrics, release
  dates, comparability breaks, and fixed-panel membership.
- `scripts/analyze.py` — descriptive frontiers, legacy calendar fits,
  measurement coverage, cluster bootstrap, leave-one-out checks, and stationary
  record null.
- `scripts/analyze_unlocks.py` — continuous score-mass records,
  censoring-aware candidate laws, rolling forecasts, cluster bootstrap,
  ablations, permutations, and matched-schedule simulations.
- `scripts/plot_paper.py` — the three compact evidence figures.

## Main limitations

- The corpus is curated rather than an exhaustive registry.
- Most dates are system release dates, not verified score-disclosure dates.
- Evaluation and reporting are selective; old benchmarks are frequently stale.
- Score thresholds and benchmarks are clustered, not independent observations.
- Native 0–100 support does not create a common semantic difficulty scale.
- Retrospective pseudo-out-of-sample results can contain future backfill
  leakage.

The decisive next step is a prospective, regularly measured fixed panel with
separate system-availability and disclosure dates.

## Primary references

- [EdgeBench paper](https://arxiv.org/abs/2607.05155) and [interactive law
  explanation](https://edge-bench.org/#lawanim)
- [Epoch Capabilities Index methodology](https://epoch.ai/data/eci-documentation/methodology)
- [SWE-bench leaderboard](https://www.swebench.com/index.html)
- [ARC Prize leaderboard](https://arcprize.org/leaderboard)
- [HLE paper](https://arxiv.org/abs/2501.14249)
