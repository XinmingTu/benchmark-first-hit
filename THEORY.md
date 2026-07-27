# An Exposure–Threshold Theory of Benchmark First Hits

This note states the theoretical interpretation of this repository. It is a
working model, not a claim that a universal law has already been established.
The current empirical object is the retrospective date on which a system
capable of crossing a fixed benchmark score threshold became available. The
system may use any model, agent, scaffold, tool configuration, or inference
budget, provided that the benchmark edition, task set, metric, and denominator
remain comparable.

The proposed connection to environment learning is precise but limited:
EdgeBench's within-run learning curves and this repository's calendar-time
frontiers can share an **effective-exposure / threshold-discovery** structure.
They do not necessarily share the same clock, learner, or causal mechanism.

## 1. Two legitimate date estimands

There are two different first-hit clocks, and they must not be described as if
they were the same.

The **public-evidence frontier** assigns a score to the date on which sufficient
evidence for that score first became public. It answers: “What did the public
record establish by date \(\tau\)?” This is the clean estimand for studying
disclosure, measurement coverage, and forecasting from information actually
available at the time.

The **retrospective capability-availability frontier** assigns a score to the
release or submission date of the system that produced it, even when the score
was published in a later back-test. It answers: “Given everything measured so
far, when did a system with this demonstrated capability become available?”
It is useful for reconstructing model and agent progress, but it can place a
score before the benchmark itself existed. Such a date is a retrospective
attribution, not a contemporaneously observed hit.

If both dates are known, then for the same result

\[
T^{\mathrm{cap}}_{bq}\le T^{\mathrm{pub}}_{bq}.
\]

The repository currently uses the capability-availability date as its primary
`date` coordinate and records the provenance in `date_basis`. It exports every
row that predates benchmark release as an explicit audit. A fully
publication-time analysis requires a separate verified disclosure date for
every retrospective result; the present corpus is not yet complete enough to
claim that estimand.

## 2. A representation identity is not a mechanism

For benchmark \(b\) and a chosen date axis \(a\in\{\mathrm{cap},\mathrm{pub}\}\),
let \(m_b^{(a)}(\tau)\in[0,1]\) be its best-so-far score by calendar date
\(\tau\), after orienting the metric so that higher is better. For a score
threshold \(q\in[0,1]\), define its assigned first-hit time

\[
T^{(a)}_{bq}
=
\inf\{\tau:m_b^{(a)}(\tau)\ge q\}.
\]

If \(Q\sim\mathrm{Uniform}(0,1)\), then

\[
\Pr_Q(T^{(a)}_{bQ}\le \tau)
=m_b^{(a)}(\tau).
\]

For a fixed panel of \(B\) equally weighted benchmarks,

\[
X^{(a)}(\tau)
=\frac{1}{B}\sum_{b=1}^{B}m_b^{(a)}(\tau)
=\Pr_{b,Q}(T^{(a)}_{bQ}\le\tau),
\]

where \(b\) is sampled uniformly from the panel. Thus the aggregate frontier is
exactly the CDF of first-hit dates for a randomly selected benchmark and score
height.

This identity is useful, but it is true for every monotone best-so-far curve.
It does not explain why the first-hit distribution should be logistic, why
progress occurred, or whether learning rather than search produced it. The
"score atoms" here are nested attainment thresholds, not necessarily semantic
skills, benchmark items, or equally important units of capability. Equal
benchmark weighting and uniform weighting over score height are explicit
measurement choices.

## 3. The effective-exposure model

Let \(A(t)\ge0\) denote cumulative **effective exposure** to opportunities for
progress. Depending on the setting, it may summarize valid attempts, feedback
informativeness, exploration novelty, retained memory, model updates, research
effort, and algorithmic efficiency. Give each score threshold a
discoverability parameter \(\lambda>0\). A simple sufficient model is

\[
\Pr(T>t\mid\lambda)=\exp[-\lambda A(t)].
\]

If the score-weighted distribution of discoverability is \(G\), the expected
unlocked fraction is

\[
\boxed{
x(t)=1-\mathbb{E}_{\lambda\sim G}
\left[e^{-\lambda A(t)}\right]
=1-\mathcal{L}_G(A(t))
},
\]

where \(\mathcal{L}_G\) is the Laplace transform of \(G\). This is a
heterogeneous first-hit or frailty model. Conditional independence is one
possible microfoundation, but it is not required for the aggregate
representation.

An especially transparent case is
\(\lambda\sim\mathrm{Exponential}(1)\):

\[
x(t)=\frac{A(t)}{1+A(t)}
=\sigma(\log A(t)),
\]

and therefore

\[
\boxed{
\frac{dx}{dt}
=
\frac{d\log A}{dt}\,x(1-x)
}.
\]

The same macroscopic equation can arise from different mechanisms. EdgeBench,
for example, derives \(x(1-x)\) from frontier expansion on a latent task graph:
unlocked score mass supplies reusable capability, while locked mass represents
remaining opportunity. Its cut-mixing and fine-granularity assumptions lead to
a logistic mean-field limit. Heterogeneous discoverability instead produces
the same form by exhausting easy thresholds before difficult ones. Aggregate
curve shape alone cannot distinguish these mechanisms.

More fundamentally, observations identify only the composition
\(1-\mathcal{L}_G(A(t))\). Without an external measure or intervention on
exposure, \(A\) and \(G\) are not separately identifiable. Almost any monotone
curve can be represented by a suitable threshold distribution and time
reparameterization.

## 4. Two clocks: interaction time and calendar time

### EdgeBench's within-run clock

EdgeBench uses elapsed interaction time \(t>0\), and fits

\[
S(t)
=
\frac{S_{\max}}
{1+(t_{\mathrm{mid}}/t)^\beta}.
\]

In the exposure model, this follows if

\[
A_{\mathrm{run}}(t)
=
\left(\frac{t}{t_{\mathrm{mid}}}\right)^\beta.
\]

Then

\[
\frac{dx}{d\log t}=\beta x(1-x).
\]

EdgeBench proposes a complementary explanation for the logarithmic clock:
when each additive increase in task-graph difficulty exposes multiplicatively
more search structure, steady raw effort reaches difficulty proportional to
\(\log t\). The paper also stresses that the smooth law is population-level:
individual tasks remain jagged, and an average of task sigmoids need not be a
sigmoid unless their midpoints and speeds are sufficiently aligned.

### The calendar clock

Let \(\tau\) be an ordinary calendar coordinate measured in years. If cumulative
effective ecosystem exposure grows approximately exponentially,

\[
A_{\mathrm{eco}}(\tau)
=
\exp[k(\tau-\tau_{\mathrm{mid}})],
\]

then the same exponential-discoverability model gives

\[
x(\tau)=\sigma[k(\tau-\tau_{\mathrm{mid}})].
\]

This is a sigmoid in **raw calendar time**, not in the logarithm of a calendar
date. Taking the logarithm of a year number would depend on an arbitrary date
origin. The substantive hypothesis is instead that the logarithm of effective
innovation exposure is approximately affine in calendar time.

That hypothesis is plausible in a regime where research compute, investment,
experimentation, and algorithmic efficiency compound, but the benchmark data
do not establish it by themselves. EdgeBench separately reports a release-date
trend in two-hour environment-learning gains on a standardized 18-task slice.
That controlled comparison is closer to measuring changing learning speed than
heterogeneous vendor-reported benchmark records.

## 5. The boundary of the learning claim

A rising best-so-far curve is not sufficient evidence of learning. An
operational learning claim requires past experience or accumulated knowledge
to causally improve the distribution of future outcomes.

Within one stateful agent run, this can be tested by holding total effort fixed
and comparing retained experience with resets. EdgeBench performs such a
comparison: continuous runs outperform independent restarts under the same
time budget. Test-Time Training to Discover similarly compares parameter
updates against matched-budget Best-of-\(N\) search.

The calendar frontier is different. Its successive records may come from
different organizations, base models, agents, scaffolds, and test-time
budgets. It can reasonably be interpreted as cumulative innovation or
ecosystem-level learning, because later systems may inherit research knowledge
from earlier work. But the observable is only a **retrospectively measured
system frontier** (or, on the alternative clock, a public-evidence frontier).
It does not show that one model learned continuously, nor does it separate
training progress, inference-time search, benchmark-specific tuning,
contamination, and reporting selection.

Continuous adaptation is not guaranteed to help. Agentic test-time training
work reports that repeated self-training text can amplify drift, with results
suggesting that its gains mainly preserve existing competence rather than
create new abilities. AgentOdyssey reports exploration collapse, memory
failures, and results consistent with catastrophic forgetting in long-horizon
agents. Effective exposure \(A(t)\) should therefore not be assumed to grow at
a constant positive rate: redundant attempts, forgetting, or uninformative
feedback can reduce \(A'(t)\).

## 6. The repeated-sampling null

Suppose a benchmark receives \(n\) evaluations whose scores are independent
draws from a stationary distribution with CDF \(F_b\). Even with no improvement
in that distribution, the maximum \(M_{b,n}\) rises:

\[
\Pr(M_{b,n}<q)=F_b(q)^n.
\]

More evaluations therefore create new records by chance. Calendar-time
first-hit curves can mix at least four processes:

\[
\text{technical progress}
+\text{more sampling opportunities}
+\text{selective publication}
+\text{changing benchmark coverage}.
\]

A credible ecosystem-learning result must exceed a repeated-sampling null.
One option is to simulate stationary score draws using the actual number and
timing of comparable evaluations. A stronger design evaluates a fixed panel
under a standardized harness at regular intervals. Analyses should retain all
comparable observations, not only frontier records, because non-record scores
help determine whether the underlying score distribution is shifting.

## 7. Measurement coverage and censoring

Let \(\pi_b(\tau)\) denote the probability that relevant new systems are
publicly and comparably evaluated on benchmark \(b\). A reduced-form public
exposure is

\[
H_b(\tau)
=
\int^\tau \pi_b(s)\,dA_{\mathrm{eco}}(s),
\qquad
x_b^{\mathrm{obs}}(\tau)
=
1-\mathcal{L}_{G_b}(H_b(\tau)).
\]

When no new comparable measurement occurs, \(\pi_b\) is effectively zero and
the observed frontier remains flat. On the public-evidence axis, that plateau
means “no new public first hit,” not “no latent capability improvement.” On the
retrospective capability axis, a future back-test may move an attributed hit
backward in time, so the historical curve is revision-prone.

The censoring semantics depend on the clock:

- On the public-evidence axis, time before benchmark release or corpus entry is
  **not at risk / NA** (left truncation), not a score of zero.
- On the retrospective capability axis, a later back-test may attribute a
  crossing to a system released before the benchmark. This is an explicitly
  flagged retrospective reconstruction, not evidence that the hit was
  observable then.
- Thresholds below the first recorded score: **left-censored**.
- Previously attained thresholds: permanently **carried forward**.
- Thresholds above the last observed frontier at the analysis cutoff:
  **right-censored**.

Carry-forward is correct on either explicitly chosen first-hit axis because an
assigned threshold crossing cannot become unattained. It would not be
sufficient for estimating the current latent capability of an untested model.
A separate freshness or active-coverage series should report the fraction of
the panel receiving recent comparable evaluations.

Coverage is also likely informative. Labs may report benchmarks on which a
model performs well; saturated benchmarks disappear; and new editions replace
old ones. Without standardized backtesting or an explicit observation model,
latent progress and evaluation propensity cannot be disentangled.

## 8. Why log-error and power laws can coexist with a sigmoid

Let \(R=1-x\) be remaining unattained score mass. Under exponential
discoverability,

\[
R(A)=\frac{1}{1+A}.
\]

For EdgeBench's \(A(t)\propto t^\beta\),

\[
x(t)\sim t^\beta \quad(t\ll t_{\mathrm{mid}}),
\qquad
1-x(t)\sim t^{-\beta}\quad(t\gg t_{\mathrm{mid}}).
\]

Power-law score growth and power-law remaining error are therefore the early
and late asymptotes of the same log-sigmoid.

If calendar exposure is exponential, \(A(\tau)\propto e^{k\tau}\), the upper
tail satisfies

\[
1-x(\tau)\sim e^{-k\tau}.
\]

Consequently, \(\log(1-x)\) is approximately linear in calendar time. A
log-error model winning over a fitted calendar sigmoid on a short, high-score
window does not necessarily select a different mechanism; it may simply fit
the sigmoid's upper tail with fewer parameters.

Other discoverability distributions imply nearby shapes. A concentrated
\(\lambda\) gives exponential survival in exposure and a Weibull curve when
\(A\propto t^\beta\). Gamma heterogeneity gives generalized Hill curves.
Normal or extreme-value threshold distributions in \(\log A\) lead to
log-probit or Gompertz forms. EdgeBench reports very similar errors for several
of these S-curves, so curve choice should rely on held-out prediction and
mechanistic tests, not small in-sample \(R^2\) differences.

## 9. Falsifiable predictions

The framework makes predictions that can fail:

1. **Exposure collapse.** Curves from systems differing only in attempt
   throughput should align when plotted against measured effective exposure.
2. **Rate intervention.** If exposure rate is multiplied by \(c\) and
   \(A\propto t^\beta\), then \(t_{\mathrm{mid}}\) should change by
   \(c^{-1/\beta}\) while \(\beta\) and the attainable ceiling remain stable.
3. **Novelty prediction.** Cumulative valid, novel, feedback-bearing attempts
   should predict progress better than wall time or raw token count. Rising
   repetition should predict a falling effective exposure rate and plateaus.
4. **Calendar-exposure prediction.** Raw-calendar logistic behavior should
   weaken if external proxies show that \(\log A_{\mathrm{eco}}\) is not close
   to linear in calendar time.
5. **Repeated-sampling test.** The observed record curve should outperform a
   stationary-score null with the same evaluation count and timing.
6. **Coverage intervention.** Regular standardized backfills of old
   benchmarks should not radically alter the inferred law. Large delayed jumps
   correlated with testing activity would indicate observation-driven results.
7. **Panel stability.** Parameters should remain reasonably stable under
   domain, benchmark-vintage, source, and leave-one-benchmark-out splits.
8. **Mechanism discrimination.** A task-graph mechanism predicts conditional
   clustering and acceleration among related score units after a neighboring
   unlock; a pure independent-discoverability mixture does not.
9. **Out-of-sample forecasting.** Fits made on an early calendar window should
   predict later first-hit mass better than linear, stationary-record, and
   flexible monotone baselines.

## 10. Estimation implications

The benchmark edition, task set, metric, denominator, and score direction must
be fixed. Model choice, scaffold, tools, and inference budget may vary because
they are part of the system frontier; protocol changes that alter the evaluated
task or the meaning of the metric are not comparable.
The date rule must also match the stated estimand. Use earliest verified score
disclosure for the public-evidence frontier; use system release or submission
date for the retrospective capability frontier, and mark later back-tests as
such. Never silently mix the two.

A frontier jump from \(s_0\) to \(s_1\) jointly records first hits for thresholds
in \((s_0,s_1]\). These thresholds and benchmarks hit by the same model release
are dependent. Uncertainty should therefore use benchmark- and release-cluster
bootstrap procedures rather than treating score points as independent.

Monthly carry-forward values are useful for plotting and for an explicitly
time-weighted loss, but they are deterministic repetitions of the same state.
Treating them as independent observations produces overly optimistic standard
errors and information criteria. Event-time or censored-survival likelihoods
are more natural.

Late benchmarks can be retained without aligning them to \(t-t_{\mathrm{first}}\):
use calendar-time models with left-truncated entry, hierarchical benchmark
effects, or fixed calendar-vintage cohorts. A fixed common panel remains a
clear descriptive alternative, provided excluded and stale benchmarks are
reported.

Finally, a fitted \(S_{\max}\) is weakly identified when the data cover only a
short upper tail. Boundary estimates, ceiling forecasts, and characteristic
times should be treated as descriptive until validated out of sample.

## 11. Claim in one sentence

> Benchmark First Hit measures the assigned crossing times of fixed score
> thresholds under cumulative technical exposure and selective measurement;
> the present primary clock is retrospective capability availability, while
> first public evidence is a distinct estimand.

This statement is compatible with environment learning, test-time training,
search, and ecosystem innovation. Distinguishing them requires interventions,
matched-budget nulls, and explicit modeling of the observation process.

## Primary sources

- Zhu et al., [*EdgeBench: Unveiling Scaling Laws of Learning from Real-World
  Environments*](https://arxiv.org/abs/2607.05155), 2026; see also the
  [official interactive explanation](https://edge-bench.org/#lawanim).
- Yuksekgonul et al.,
  [*Learning to Discover at Test Time*](https://arxiv.org/abs/2601.16175),
  2026.
- Wang et al.,
  [*No Time Like the Present: Agentic Test-Time Training for LLM
  Agents*](https://arxiv.org/abs/2607.03441), 2026.
- Zhang et al.,
  [*AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time
  Continual Learning Agents*](https://arxiv.org/abs/2606.24893), 2026.
- Kaplan et al.,
  [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361),
  2020.
