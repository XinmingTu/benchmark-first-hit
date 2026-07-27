# Theory: First-Hit Diffusion Under Selective Measurement

This note separates three things that a progress curve can otherwise blur:

1. an exact representation of a benchmark frontier;
2. a candidate model of latent capability unlocks; and
3. the selective process by which those unlocks become measured.

The representation is a theorem. The exposure model is a hypothesis. A good
curve fit is not, by itself, evidence for the hypothesis.

## 1. The empirical object

Fix a benchmark edition, evaluated task set, metric, denominator, and score
direction. Any base model, agent scaffold, tool stack, ensemble, or
test-time-compute budget may set its record. This defines an **any-system
frontier**, not a base-model leaderboard.

Let a frozen corpus \(\mathcal C\) contain comparable score reports. Three
first-hit times are conceptually distinct for benchmark \(b\) and score
threshold \(q\):

\[
\begin{aligned}
U_{bq}
&=\text{first time any available system could in fact attain }q,\\
R^{\mathcal C}_{bq}
&=\text{earliest system-availability date assigned by the frozen corpus},\\
T^{\mathcal C}_{bq}
&=\text{first date on which the corpus says attainment was publicly disclosed}.
\end{aligned}
\]

Under complete and correctly dated evidence,

\[
U_{bq}\le R^{\mathcal C}_{bq}\le T^{\mathcal C}_{bq}.
\]

Their statistical meanings differ:

- \(U\) is a latent capability unlock. Selective public results do not identify
  it.
- \(R^{\mathcal C}\) is a retrospective corpus attribution. A later back-test
  can move it backward.
- \(T^{\mathcal C}\) is a public-evidence time. It is the appropriate clock for
  genuine real-time forecasting, but it requires a verified disclosure date.

The present dataset primarily records \(R^{\mathcal C}\), with `date_basis`
marking release, submission, evaluation, publication, and retrospective
attributions. It must therefore be described as a
**corpus-attributed retrospective frontier**, not as a point estimate of
latent \(U\).

## 2. The first-hit representation theorem

Let \(m_b^{\mathcal C}(\tau)\in[0,1]\) be benchmark \(b\)'s corpus frontier by
date \(\tau\). Draw a threshold \(Q\sim\mathrm{Uniform}(0,1)\), and define

\[
R^{\mathcal C}_{bQ}
=
\inf\{\tau:m_b^{\mathcal C}(\tau)\ge Q\}.
\]

Then

\[
\begin{aligned}
\Pr_Q(R^{\mathcal C}_{bQ}\le\tau)
&=\int_0^1
\mathbf 1\{q\le m_b^{\mathcal C}(\tau)\}\,dq\\
&=m_b^{\mathcal C}(\tau).
\end{aligned}
\]

For benchmark weights \(w_b\ge0\), \(\sum_b w_b=1\),

\[
\boxed{
X^{\mathcal C}(\tau)
=
\sum_b w_bm_b^{\mathcal C}(\tau)
=
\Pr_{B,Q}(R^{\mathcal C}_{BQ}\le\tau)
}.
\]

Thus an equal-benchmark frontier is exactly the empirical CDF of assigned
first-hit dates for a uniformly selected benchmark and score height. This is
the project's cleanest result. It requires no sigmoid, exposure model, or
learning interpretation.

The normalization is deliberately modest. Mapping each published score to
\([0,1]\) makes every benchmark contribute one unit of **score mass**; it does
not make one percentage point equally difficult or equally valuable across
benchmarks. An overlap-calibrated IRT index can estimate a different latent
capability scale, as in the [Epoch Capabilities
Index](https://epoch.ai/data/eci-documentation/methodology), but it requires
reliably matched system configurations across benchmarks and answers a
different question.

## 3. Continuous score mass and censoring

Suppose a benchmark's genuine corpus frontier is

\[
(d_1,s_1),\ldots,(d_K,s_K),
\qquad
0<s_1<\cdots<s_K\le1.
\]

Its one unit of score mass decomposes exactly into:

\[
\underbrace{s_1}_{\text{left-censored/prevalent}}
+
\underbrace{\sum_{j=2}^{K}(s_j-s_{j-1})}_{\text{dated frontier jumps}}
+
\underbrace{(1-s_K)}_{\text{unresolved/measurement-censored}}
=1.
\]

- The first recorded score is prevalent mass: those thresholds were already
  attained at the first reliable observation, but their earlier dates are not
  known.
- A later jump of size \(s_j-s_{j-1}\) is tied score mass assigned to date
  \(d_j\). It is one clustered record event, not thousands of independent
  0.1-point observations.
- Remaining mass is unresolved. It is right-censored only to the extent that
  the relevant source was actually audited through a stated cutoff. A stale
  benchmark is measurement-censored, not evidence of no latent progress.

For a public-evidence analysis, time before benchmark release is not at risk.
For a retrospective analysis, an older system may be back-tested and assigned
a date before the benchmark existed; that is a visible backdating operation,
not a contemporaneous hit.

A previous public low score ordinarily does not prove that all unreported
systems were below a threshold. Consequently, intervals
\((d_{j-1},d_j]\) are valid latent-capability censoring intervals only under a
systematic inspection protocol. The main corpus generally supports exact
**assigned record dates**, not exact latent unlock dates.

This decomposition resolves the expanding-panel problem without aligning every
benchmark to \(t-t_{\mathrm{first}}\). A new benchmark's first score enters as
left-censored prevalent mass rather than a newly created calendar-time jump.
Subsequent events remain on real calendar time.

## 4. A capability-threshold bridge

An IRT-style response model makes the score-threshold interpretation explicit.
Suppose configuration \(m\) has capability \(C_m\), and benchmark \(b\) has
difficulty \(D_b\) and discrimination \(\alpha_b>0\):

\[
p_{mb}
=
\sigma[\alpha_b(C_m-D_b)].
\]

Attaining score threshold \(q\) is equivalent to crossing the latent difficulty

\[
\theta_{bq}
=
D_b+\frac{\operatorname{logit}(q)}{\alpha_b}.
\]

Let

\[
C^*(t)=\max_{m:r_m\le t}C_m
\]

be the best capability among configurations available by \(t\). Then

\[
\boxed{
S(t)
=
\Pr_w[\theta\le C^*(t)]
=
G_\theta(C^*(t))
}.
\]

This gives a clean bridge:

- benchmark score mass is a distribution of capability thresholds;
- model-and-agent progress supplies a frontier capability clock \(C^*(t)\);
- the aggregate first-hit curve is the threshold CDF evaluated at that clock.

If \(G_\theta\) is logistic and \(C^*(t)\) is affine in calendar time, the
result is a calendar logistic. If \(C^*(t)\) is affine in log interaction time,
the result has EdgeBench's log-time form. Heterogeneous discrimination, score
noise, and mixtures of benchmark families generally produce a mixture rather
than one exact sigmoid.

This bridge is theoretical, not a license to merge unlike evaluations. A
practical IRT fit needs the same system configuration measured across several
benchmarks. In the current corpus, strict identities split model and agent
evaluations into disconnected components; relaxing identity would conflate
tools, scaffolds, tracks, and inference budgets. IRT is therefore a useful
local sensitivity analysis and future calibration route, not the primary
index.

## 5. A latent exposure-threshold model

Let \(A(t)\ge0\) be cumulative effective exposure to progress opportunities.
It can summarize valid experiments, informative feedback, retained experience,
training effort, search, and algorithmic efficiency. Give threshold \(j\) a
discoverability \(\lambda_j>0\), and posit

\[
\Pr(U_j>t\mid\lambda_j)=\exp[-\lambda_jA(t)].
\]

If threshold discoverabilities follow distribution \(G\), the expected latent
unlocked fraction is

\[
\boxed{
F_U(t)
=1-\mathbb E_{\lambda\sim G}e^{-\lambda A(t)}
=1-\mathcal L_G(A(t))
},
\]

where \(\mathcal L_G\) is the Laplace transform of \(G\).

For \(\lambda\sim\mathrm{Exponential}(1)\),

\[
F_U(t)
=\frac{A(t)}{1+A(t)}
=\sigma(\log A(t)),
\]

so

\[
\frac{dF_U}{dt}
=
\frac{d\log A}{dt}F_U(1-F_U).
\]

This supplies one microfoundation for a sigmoid: easy or discoverable score
mass is exhausted before difficult mass. It is not unique. Frontier expansion
on a well-mixed task graph, a distribution of IRT-like difficulty thresholds,
or other heterogeneous first-passage processes can produce the same aggregate
shape.

## 6. Unlock and detection are different processes

Let \(\rho_b(t)\) be the hazard that an already unlocked threshold on benchmark
\(b\) is comparably tested and publicly reported. A threshold moves through

\[
\text{locked}
\longrightarrow
\text{unlocked but undetected}
\longrightarrow
\text{detected}.
\]

Conditional on latent unlock at time \(u\), the public-detection CDF is

\[
\boxed{
F_T(t)
=
\int_{-\infty}^{t}
\left[
1-\exp\left(-\int_u^t\rho_b(s)\,ds\right)
\right]dF_U(u)
}.
\]

Immediate and complete measurement makes \(T=U\). Low or declining
\(\rho_b(t)\) delays and flattens the observed curve even if latent progress
continues. Selective reporting makes \(\rho_b\) depend on the unreported score
and invalidates ordinary independent-censoring assumptions.

Retrospective attribution \(R^{\mathcal C}\) partially backdates a detection to
the producing system's release or submission date. It can reduce apparent
reporting delay, but it does not recover untested systems and is revision-prone
as \(\mathcal C\) grows.

## 7. EdgeBench and calendar time

[EdgeBench](https://edge-bench.org/#lawanim) fits within-run interaction time
\(t>0\) with

\[
S(t)
=
\frac{S_{\max}}
{1+(t_{\mathrm{mid}}/t)^\beta}.
\]

In the exposure model this is the special case

\[
A_{\mathrm{run}}(t)
=
\left(\frac{t}{t_{\mathrm{mid}}}\right)^\beta,
\qquad
\frac{dx}{d\log t}=\beta x(1-x).
\]

EdgeBench gives a complementary task-graph derivation and emphasizes that the
smooth law is population-level: individual tasks are jagged, and mixtures of
heterogeneous task curves need not themselves be exactly logistic. Its
matched-budget stateful-versus-reset experiment also provides evidence that
retained experience contributes beyond independent repeated attempts.

Calendar time \(\tau\) has no natural logarithmic origin. A raw-calendar
sigmoid instead requires

\[
A_{\mathrm{eco}}(\tau)
\propto
\exp[k(\tau-\tau_{\mathrm{mid}})],
\]

which yields

\[
x(\tau)=\sigma[k(\tau-\tau_{\mathrm{mid}})].
\]

The substantive claim is therefore not “take the logarithm of the year.” It is
that the logarithm of effective ecosystem exposure is approximately affine in
ordinary calendar time. That claim must predict held-out periods or agree with
an external exposure proxy; it cannot be inferred from an attractive fit.

## 8. What the curve cannot identify

The observed frontier alone does not separately identify:

- latent unlock \(U\), retrospective attribution \(R^{\mathcal C}\), and public
  detection \(T^{\mathcal C}\);
- exposure \(A(t)\) and discoverability distribution \(G\);
- technical progress and an increasing number of evaluation attempts;
- capability change and changing benchmark coverage;
- a finite attainable ceiling and a slowly moving upper tail;
- a shared law and a mixture of benchmark-specific laws;
- base-model progress, scaffold improvement, tools, inference compute,
  contamination, and benchmark-specific optimization.

The exposure representation is especially underidentified. For any monotone
\(0<X(t)<1\), defining

\[
A(t)=\frac{X(t)}{1-X(t)}
\]

makes the exponential-discoverability model fit exactly. A common clock becomes
a scientific hypothesis only when it is fixed on one subset and predicts a
different benchmark family, vintage, source, or future period.

Likewise, rising maxima do not require a changing score distribution. If
\(n\) scores are independent draws from a stationary CDF \(H_b\),

\[
\Pr(M_{b,n}<q)=H_b(q)^n.
\]

More attempts alone produce new records. Repeated-sampling and
schedule-preserving nulls are therefore mandatory. This mechanism is
empirically important in large-sample inference; see [Large Language
Monkeys](https://arxiv.org/abs/2407.21787) and work on [optimal test-time
compute](https://openreview.net/pdf?id=4FWAwZtd2n).

## 9. Estimation as a clustered event-time problem

For a parametric assigned-time CDF \(F_\theta\), continuous score mass suggests
the weighted likelihood

\[
\ell(\theta)
=
\sum_i w_i
\begin{cases}
\log F_\theta(R_i), & \text{left-censored},\\
\log f_\theta(t_i), & \text{exact assigned event},\\
\log[1-F_\theta(L_i)], & \text{right-censored}.
\end{cases}
\]

Interval terms
\(\log[F_\theta(R_i)-F_\theta(L_i)]\) are appropriate only when both bounds have
valid inspection semantics. Nonparametric interval-censored estimation can use
the [Turnbull
framework](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1976.tb01597.x).

This likelihood is best viewed as a score-mass quasi-likelihood. Thresholds
crossed by the same jump are tied, benchmarks share tasks and training data,
and one model release can move many frontiers. Uncertainty must cluster by
benchmark or family and, where possible, by system release. Monthly
carry-forward markers are deterministic state displays and must not be counted
as independent events.

Candidate calendar laws should include at least logistic, probit, Gompertz,
upper-tail/log-error, a fitted-ceiling or cure model, and no-change. In-sample
\(R^2\) and AICc are descriptive. The primary comparison is rolling-origin
90/180-day prediction under a predeclared loss, with benchmark-cluster
uncertainty.

Because the current corpus lacks complete disclosure dates, such backtests are
**retrospective pseudo-out-of-sample**: future backfills can leak into the
reconstructed past. A true prospective test requires frozen corpus snapshots
or a `known_by_date` field.

## 10. Tests that can falsify the theory

A useful exposure law should survive tests it was not constructed to pass:

1. **Future prediction.** An early fit should beat no-change and upper-tail
   baselines on later score mass.
2. **Clock transfer.** One calendar clock should transfer across benchmark
   vintage, family, and source without refitting its functional form.
3. **Schedule null.** The observed gain should exceed stationary-score
   simulations with matched evaluation dates and counts.
4. **Measurement simulation.** Selective reporting, saturation retirement, and
   retrospective backdating alone should not routinely reproduce the observed
   law.
5. **Jump robustness.** Results should not disappear after removing the largest
   release-level jumps or using one edition per benchmark family.
6. **Coverage intervention.** Regular backfills of stale benchmarks should not
   radically rewrite the inferred curve.
7. **Exposure intervention.** Multiplying measured valid attempt throughput
   should shift the time scale as predicted while preserving shape.
8. **Learning control.** At fixed total compute, retained-state systems should
   outperform matched resets before the calendar curve is called a learning
   curve in the strict causal sense.

Failure is informative. If several S-curves fit in sample but cannot beat
no-change prospectively, the correct conclusion is a smooth retrospective
description, not a scaling law.

## 11. A prospective protocol

The clean experiment is simple:

1. freeze a fixed panel and exact evaluation protocols;
2. evaluate a declared set of frontier systems on a regular schedule;
3. store system availability, evaluation, and public disclosure dates
   separately;
4. retain non-record scores as well as records;
5. publish a dated corpus snapshot and preregister 6- and 12-month forecasts;
6. report both the raw score-mass index and measurement freshness.

This design identifies a public first-hit process far more cleanly. Matched
stateful/reset or fixed-compute interventions are still required to identify
learning as the causal mechanism.

## 12. Claim in one sentence

> Benchmark First Hit is an event-time index of when fixed score mass first
> enters an observed any-system frontier; an exposure-threshold model can
> unify within-run and calendar curves, but selective measurement determines
> how much latent capability becomes visible.

## Primary sources

- Zhu et al., [*EdgeBench: Unveiling Scaling Laws of Learning from Real-World
  Environments*](https://arxiv.org/abs/2607.05155), 2026; [official interactive
  explanation](https://edge-bench.org/#lawanim).
- Sevilla et al., [*A Rosetta Stone for AI
  Benchmarks*](https://arxiv.org/abs/2512.00193), 2025; [ECI
  methodology](https://epoch.ai/data/eci-documentation/methodology).
- Snell et al., [*Scaling LLM Test-Time Compute
  Optimally*](https://openreview.net/pdf?id=4FWAwZtd2n), 2024.
- Brown et al., [*Large Language Monkeys*](https://arxiv.org/abs/2407.21787),
  2024.
- Turnbull, [*The Empirical Distribution Function with Arbitrarily Grouped,
  Censored and Truncated
  Data*](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1976.tb01597.x),
  1976.
- Liang, Lu, and Ying, [*Joint Modeling and Analysis of Longitudinal Data with
  Informative Observation
  Times*](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1541-0420.2008.01104.x),
  2009.
- Blum and Hardt, [*The Ladder: A Reliable Leaderboard for Machine
  Learning Competitions*](https://proceedings.mlr.press/v37/blum15.html), 2015.
