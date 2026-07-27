#!/usr/bin/env python3
"""Censoring-aware score-mass first-hit analysis.

This module treats every benchmark's native 0--100 score range as one unit of
``score mass``.  For a benchmark with frontier scores s_1 < ... < s_K:

* s_1 / 100 is left-censored at the first comparable observation;
* (s_j - s_{j-1}) / 100 is an observed tied first-hit event at date j; and
* (100 - s_K) / 100 is right-censored at the latest comparable observation.

Consequently every benchmark contributes total weight one, independent of its
number of observations.  The event dates are dates of *observed first hits*.
The previous evaluation date is exported as an interval-candidate diagnostic,
but it is not used as, or claimed to be, a lower bound on a latent capability
unlock time.

The fitted distributions are weighted composite-likelihood descriptions of
the score-mass first-hit record.  They are not iid likelihoods: thresholds
within a benchmark are dependent, publication schedules are selective, and
several benchmark editions share ancestry.  Benchmark-cluster bootstrap,
schedule-matched simulations, rolling-origin backtests, jump ablations, and a
time/jump permutation null make those limitations empirically visible.

Run from the project root:

    python scripts/analyze_unlocks.py

No plots are created.  All outputs are CSV files under ``output/`` by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr

from analyze import load_data, make_frontiers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output"
EPOCH = pd.Timestamp("2018-01-01")
YEAR_DAYS = 365.25
EPS = 1e-12
MODEL_NAMES = (
    "logistic",
    "probit",
    "gompertz",
    "exponential_tail",
    "cure_logistic",
)
INTERVAL_CAVEAT = (
    "diagnostic observation interval only; not an identified latent-unlock "
    "lower bound"
)
BACKTEST_CAVEAT = (
    "final-corpus retrospective rolling-origin backtest; panel construction, "
    "source discovery, and capability-date assignments are not vintage-frozen"
)


@dataclass(frozen=True)
class ModelSpec:
    """A compact representation of one first-hit distribution."""

    name: str
    n_params: int
    initial: Callable[[pd.DataFrame], np.ndarray]
    bounds: Callable[[pd.DataFrame], list[tuple[float, float]]]
    distribution: Callable[
        [np.ndarray, np.ndarray, pd.DataFrame], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
    report: Callable[[np.ndarray, pd.DataFrame], dict[str, object]]


def _date_to_years(value: pd.Series | pd.DatetimeIndex | pd.Timestamp) -> np.ndarray:
    """Convert real calendar dates to years since a fixed, global epoch."""
    if isinstance(value, pd.Timestamp):
        return np.asarray([(value - EPOCH).total_seconds() / (YEAR_DAYS * 86400)])
    converted = pd.to_datetime(value)
    difference = converted - EPOCH
    seconds = (
        difference.dt.total_seconds()
        if isinstance(difference, pd.Series)
        else difference.total_seconds()
    )
    return np.asarray(
        seconds / (YEAR_DAYS * 86400), dtype=float
    )


def _years_to_date(value: float) -> str:
    date = EPOCH + pd.to_timedelta(float(value) * YEAR_DAYS, unit="D")
    return pd.Timestamp(date).date().isoformat()


def _daily_best(
    obs: pd.DataFrame,
    benchmark_ids: Iterable[str],
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return one best eligible score at each benchmark/date."""
    ids = set(benchmark_ids)
    selected = obs.loc[
        (obs["frontier_eligible"] == 1) & obs["benchmark_id"].isin(ids)
    ].copy()
    if cutoff is not None:
        selected = selected.loc[selected["date"] <= cutoff]
    if selected.empty:
        return selected
    idx = selected.groupby(["benchmark_id", "date"])["edge_score"].idxmax()
    return selected.loc[idx].sort_values(["benchmark_id", "date"]).reset_index(drop=True)


def build_mass_records(
    obs: pd.DataFrame,
    benchmark_ids: Iterable[str],
    cutoff: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct unit-mass censor/event records and interval diagnostics.

    Right censoring is attached to the latest comparable evaluation, not to the
    analysis cutoff: an unmeasured calendar interval is not evidence that a
    threshold remained locked.
    """
    daily = _daily_best(obs, benchmark_ids, cutoff=cutoff)
    records: list[dict[str, object]] = []
    intervals: list[dict[str, object]] = []
    for benchmark_id, group in daily.groupby("benchmark_id", sort=False):
        group = group.sort_values("date").copy()
        group["frontier_score"] = group["edge_score"].cummax()
        changed = group["frontier_score"].gt(
            group["frontier_score"].shift(fill_value=-np.inf)
        )
        frontier = group.loc[changed].copy()
        if frontier.empty:
            continue

        first = frontier.iloc[0]
        first_score = float(np.clip(first["frontier_score"], 0, 100))
        common = {
            "benchmark_id": benchmark_id,
            "benchmark_name": first["benchmark_name"],
            "category": first["category"],
            "domain": first["domain"],
        }
        if first_score > 0:
            records.append(
                {
                    **common,
                    "record_type": "left_censored",
                    "event_date": first["date"],
                    "mass": first_score / 100.0,
                    "score_lower": 0.0,
                    "score_upper": first_score,
                    "previous_frontier_score": 0.0,
                    "frontier_score": first_score,
                }
            )
            intervals.append(
                {
                    **common,
                    "record_type": "left_censored",
                    "event_date": first["date"],
                    "interval_candidate_lower": pd.NaT,
                    "interval_candidate_upper": first["date"],
                    "interval_width_days": np.nan,
                    "mass": first_score / 100.0,
                    "interpretation": INTERVAL_CAVEAT,
                }
            )

        previous_score = first_score
        daily_dates = group["date"].to_numpy(dtype="datetime64[ns]")
        for row in frontier.iloc[1:].itertuples(index=False):
            score = float(np.clip(row.frontier_score, 0, 100))
            delta = max(0.0, score - previous_score)
            if delta <= 1e-12:
                continue
            event_date = pd.Timestamp(row.date)
            earlier = daily_dates[daily_dates < np.datetime64(event_date)]
            previous_eval = pd.Timestamp(earlier[-1]) if len(earlier) else pd.NaT
            records.append(
                {
                    **common,
                    "record_type": "exact_tied_event",
                    "event_date": event_date,
                    "mass": delta / 100.0,
                    "score_lower": previous_score,
                    "score_upper": score,
                    "previous_frontier_score": previous_score,
                    "frontier_score": score,
                }
            )
            intervals.append(
                {
                    **common,
                    "record_type": "exact_tied_event",
                    "event_date": event_date,
                    "interval_candidate_lower": previous_eval,
                    "interval_candidate_upper": event_date,
                    "interval_width_days": (
                        (event_date - previous_eval).days
                        if pd.notna(previous_eval)
                        else np.nan
                    ),
                    "mass": delta / 100.0,
                    "interpretation": INTERVAL_CAVEAT,
                }
            )
            previous_score = score

        last_eval = pd.Timestamp(group["date"].max())
        remaining = max(0.0, 1.0 - previous_score / 100.0)
        if remaining > 1e-12:
            records.append(
                {
                    **common,
                    "record_type": "right_censored",
                    "event_date": last_eval,
                    "mass": remaining,
                    "score_lower": previous_score,
                    "score_upper": 100.0,
                    "previous_frontier_score": previous_score,
                    "frontier_score": previous_score,
                }
            )
            intervals.append(
                {
                    **common,
                    "record_type": "right_censored",
                    "event_date": last_eval,
                    "interval_candidate_lower": last_eval,
                    "interval_candidate_upper": pd.NaT,
                    "interval_width_days": np.nan,
                    "mass": remaining,
                    "interpretation": INTERVAL_CAVEAT,
                }
            )

    result = pd.DataFrame(records)
    diagnostic = pd.DataFrame(intervals)
    if result.empty:
        return result, diagnostic
    totals = result.groupby("benchmark_id")["mass"].sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-9):
        raise ValueError(f"Score mass does not sum to one:\n{totals}")
    result["event_date"] = pd.to_datetime(result["event_date"])
    result["time_years"] = _date_to_years(result["event_date"])
    result["fit_weight"] = result["mass"]
    diagnostic["event_date"] = pd.to_datetime(diagnostic["event_date"])
    diagnostic["interval_candidate_lower"] = pd.to_datetime(
        diagnostic["interval_candidate_lower"]
    )
    diagnostic["interval_candidate_upper"] = pd.to_datetime(
        diagnostic["interval_candidate_upper"]
    )
    return (
        result.sort_values(["benchmark_id", "event_date", "record_type"]).reset_index(
            drop=True
        ),
        diagnostic.sort_values(
            ["benchmark_id", "event_date", "record_type"]
        ).reset_index(drop=True),
    )


def define_panels(
    meta: pd.DataFrame, frontiers: pd.DataFrame
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Define fixed, all-observed, and real-calendar benchmark vintages."""
    observed_ids = set(frontiers["benchmark_id"].unique())
    fixed = [
        value
        for value in meta.loc[meta["include_in_composite"] == 1, "benchmark_id"]
        if value in observed_ids
    ]
    all_observed = [
        value for value in meta["benchmark_id"] if value in observed_ids
    ]
    release_year = meta.set_index("benchmark_id")["benchmark_release_date"].dt.year
    panels: dict[str, list[str]] = {
        "fixed31": fixed,
        "all44": all_observed,
        "vintage_pre2023": [
            value for value in all_observed if int(release_year[value]) <= 2022
        ],
        "vintage_2023": [
            value for value in all_observed if int(release_year[value]) == 2023
        ],
        "vintage_2024": [
            value for value in all_observed if int(release_year[value]) == 2024
        ],
        "vintage_2025plus": [
            value for value in all_observed if int(release_year[value]) >= 2025
        ],
    }
    names = meta.set_index("benchmark_id")["benchmark_name"].to_dict()
    membership = []
    for panel, ids in panels.items():
        for benchmark_id in ids:
            row = meta.loc[meta["benchmark_id"] == benchmark_id].iloc[0]
            membership.append(
                {
                    "panel": panel,
                    "benchmark_id": benchmark_id,
                    "benchmark_name": names[benchmark_id],
                    "benchmark_release_date": row["benchmark_release_date"],
                    "release_vintage_year": int(row["benchmark_release_date"].year),
                    "include_in_original_fixed_panel": int(
                        row["include_in_composite"]
                    ),
                    "panel_benchmark_count": len(ids),
                    "time_alignment": "real_calendar_no_relative_alignment",
                }
            )
    return panels, pd.DataFrame(membership)


def _base_initial(records: pd.DataFrame) -> tuple[float, float, float, float]:
    times = records["time_years"].to_numpy(dtype=float)
    weights = records["fit_weight"].to_numpy(dtype=float)
    order = np.argsort(times)
    cumulative = np.cumsum(weights[order])
    midpoint = float(times[order[np.searchsorted(cumulative, cumulative[-1] / 2)]])
    span = max(float(np.ptp(times)), 0.5)
    scale = max(span / 4.0, 0.15)
    return midpoint, scale, float(times.min()), span


def _common_bounds(records: pd.DataFrame, n_params: int = 2) -> list[tuple[float, float]]:
    _, _, lower, span = _base_initial(records)
    upper = float(records["time_years"].max())
    result = [(lower - 5 * span, upper + 5 * span), (math.log(0.02), math.log(20))]
    if n_params == 3:
        result.append((-8.0, 8.0))
    return result


def _logistic_distribution(
    theta: np.ndarray, time: np.ndarray, _: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    midpoint, log_scale = theta[:2]
    scale = math.exp(float(log_scale))
    value = expit((time - midpoint) / scale)
    density = value * (1 - value) / scale
    return value, density, 1 - value


def _probit_distribution(
    theta: np.ndarray, time: np.ndarray, _: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    midpoint, log_scale = theta[:2]
    scale = math.exp(float(log_scale))
    z = (time - midpoint) / scale
    value = ndtr(z)
    density = np.exp(-0.5 * np.square(z)) / (math.sqrt(2 * math.pi) * scale)
    return value, density, 1 - value


def _gompertz_distribution(
    theta: np.ndarray, time: np.ndarray, _: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    midpoint, log_scale = theta[:2]
    scale = math.exp(float(log_scale))
    z = np.clip((time - midpoint) / scale, -40, 20)
    exp_z = np.exp(z)
    survival = np.exp(-exp_z)
    value = 1 - survival
    density = exp_z * survival / scale
    return value, density, survival


def _exponential_distribution(
    theta: np.ndarray, time: np.ndarray, records: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, minimum, _ = _base_initial(records)
    onset = minimum - math.exp(float(theta[0]))
    rate = math.exp(float(theta[1]))
    elapsed = np.maximum(time - onset, 0.0)
    survival = np.exp(-rate * elapsed)
    value = 1 - survival
    density = np.where(time > onset, rate * survival, 0.0)
    return value, density, survival


def _cure_logistic_distribution(
    theta: np.ndarray, time: np.ndarray, records: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base, density, _ = _logistic_distribution(theta[:2], time, records)
    unlockable = expit(float(theta[2]))
    value = unlockable * base
    density = unlockable * density
    return value, density, 1 - value


def _standard_report(
    theta: np.ndarray, _: pd.DataFrame
) -> dict[str, object]:
    return {
        "midpoint_years_from_epoch": float(theta[0]),
        "midpoint_date": _years_to_date(float(theta[0])),
        "scale_years": math.exp(float(theta[1])),
        "rate_per_year": np.nan,
        "onset_date": "",
        "unlockable_fraction": 1.0,
    }


def _exponential_report(
    theta: np.ndarray, records: pd.DataFrame
) -> dict[str, object]:
    _, _, minimum, _ = _base_initial(records)
    onset = minimum - math.exp(float(theta[0]))
    return {
        "midpoint_years_from_epoch": np.nan,
        "midpoint_date": "",
        "scale_years": np.nan,
        "rate_per_year": math.exp(float(theta[1])),
        "onset_date": _years_to_date(onset),
        "unlockable_fraction": 1.0,
    }


def _cure_report(theta: np.ndarray, records: pd.DataFrame) -> dict[str, object]:
    result = _standard_report(theta, records)
    result["unlockable_fraction"] = float(expit(theta[2]))
    return result


def model_specs() -> dict[str, ModelSpec]:
    """Return all candidate model specifications."""

    def base_initial(records: pd.DataFrame) -> np.ndarray:
        midpoint, scale, _, _ = _base_initial(records)
        return np.asarray([midpoint, math.log(scale)])

    def exp_initial(records: pd.DataFrame) -> np.ndarray:
        _, scale, _, _ = _base_initial(records)
        return np.asarray([math.log(max(scale, 0.1)), math.log(1 / scale)])

    def cure_initial(records: pd.DataFrame) -> np.ndarray:
        return np.r_[base_initial(records), 2.2]

    return {
        "logistic": ModelSpec(
            "logistic",
            2,
            base_initial,
            lambda r: _common_bounds(r),
            _logistic_distribution,
            _standard_report,
        ),
        "probit": ModelSpec(
            "probit",
            2,
            base_initial,
            lambda r: _common_bounds(r),
            _probit_distribution,
            _standard_report,
        ),
        "gompertz": ModelSpec(
            "gompertz",
            2,
            base_initial,
            lambda r: _common_bounds(r),
            _gompertz_distribution,
            _standard_report,
        ),
        "exponential_tail": ModelSpec(
            "exponential_tail",
            2,
            exp_initial,
            lambda r: [
                (math.log(0.01), math.log(max(30.0, 10 * _base_initial(r)[3]))),
                (math.log(0.002), math.log(50.0)),
            ],
            _exponential_distribution,
            _exponential_report,
        ),
        "cure_logistic": ModelSpec(
            "cure_logistic",
            3,
            cure_initial,
            lambda r: _common_bounds(r, n_params=3),
            _cure_logistic_distribution,
            _cure_report,
        ),
    }


def _composite_nll(
    theta: np.ndarray, records: pd.DataFrame, spec: ModelSpec
) -> float:
    time = records["time_years"].to_numpy(dtype=float)
    value, density, survival = spec.distribution(theta, time, records)
    record_type = records["record_type"].to_numpy()
    contribution = np.empty(len(records), dtype=float)
    contribution[record_type == "left_censored"] = value[
        record_type == "left_censored"
    ]
    contribution[record_type == "exact_tied_event"] = density[
        record_type == "exact_tied_event"
    ]
    contribution[record_type == "right_censored"] = survival[
        record_type == "right_censored"
    ]
    contribution = np.clip(contribution, EPS, None)
    weights = records["fit_weight"].to_numpy(dtype=float)
    result = -float(np.sum(weights * np.log(contribution)))
    return result if np.isfinite(result) else 1e100


def fit_model(records: pd.DataFrame, model_name: str) -> dict[str, object]:
    """Fit one weighted composite-likelihood first-hit distribution."""
    spec = model_specs()[model_name]
    initial = spec.initial(records)
    starts = [initial]
    shifted = initial.copy()
    shifted[0] = np.clip(
        shifted[0] + 0.25 * _base_initial(records)[3],
        spec.bounds(records)[0][0],
        spec.bounds(records)[0][1],
    )
    starts.append(shifted)
    best = None
    for start in starts:
        fitted = minimize(
            _composite_nll,
            start,
            args=(records, spec),
            method="L-BFGS-B",
            bounds=spec.bounds(records),
            options={"maxiter": 500, "ftol": 1e-11},
        )
        if best is None or fitted.fun < best.fun:
            best = fitted
    assert best is not None
    nll = float(best.fun)
    total_weight = float(records["fit_weight"].sum())
    report = spec.report(np.asarray(best.x), records)
    return {
        "model": model_name,
        "converged": int(bool(best.success) and np.isfinite(nll)),
        "optimizer_message": str(best.message),
        "weighted_nll": nll,
        "weighted_aic": 2 * spec.n_params + 2 * nll,
        "n_parameters": spec.n_params,
        "benchmark_count": int(records["benchmark_id"].nunique()),
        "record_count": len(records),
        "total_benchmark_weight": total_weight,
        "exact_event_mass": float(
            records.loc[
                records["record_type"] == "exact_tied_event", "fit_weight"
            ].sum()
        ),
        "left_censored_mass": float(
            records.loc[
                records["record_type"] == "left_censored", "fit_weight"
            ].sum()
        ),
        "right_censored_mass": float(
            records.loc[
                records["record_type"] == "right_censored", "fit_weight"
            ].sum()
        ),
        **report,
        "_theta": np.asarray(best.x),
    }


def predict_model(
    fit: dict[str, object], records: pd.DataFrame, dates: pd.Series | pd.DatetimeIndex
) -> np.ndarray:
    """Evaluate a fitted first-hit CDF on real calendar dates."""
    spec = model_specs()[str(fit["model"])]
    time = _date_to_years(pd.to_datetime(dates))
    value, _, _ = spec.distribution(np.asarray(fit["_theta"]), time, records)
    return np.clip(value, 0, 1)


def fit_panels(
    obs: pd.DataFrame, panels: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit all models and export monthly prediction curves for every panel."""
    fit_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    mass_by_panel: dict[str, pd.DataFrame] = {}
    for panel, ids in panels.items():
        records, _ = build_mass_records(obs, ids)
        if records.empty or records["benchmark_id"].nunique() < 2:
            continue
        mass_by_panel[panel] = records
        fits = [fit_model(records, model_name) for model_name in MODEL_NAMES]
        valid_aic = {
            str(fit["model"]): float(fit["weighted_aic"])
            for fit in fits
            if int(fit["converged"]) == 1
        }
        winner = min(valid_aic, key=valid_aic.get) if valid_aic else ""
        grid = pd.Series(
            pd.date_range(
                records["event_date"].min(),
                records["event_date"].max() + pd.Timedelta(days=180),
                freq="30D",
            )
        )
        for fit in fits:
            clean = {key: value for key, value in fit.items() if key != "_theta"}
            clean["panel"] = panel
            clean["aic_winner"] = int(fit["model"] == winner)
            clean["likelihood_type"] = (
                "weighted composite likelihood: left-CDF, exact-density, "
                "right-survival; time unit years"
            )
            fit_rows.append(clean)
            values = predict_model(fit, records, grid)
            for date, value in zip(grid, values):
                prediction_rows.append(
                    {
                        "panel": panel,
                        "model": fit["model"],
                        "date": date,
                        "predicted_unlocked_mass": float(value),
                        "predicted_score_0_100": float(100 * value),
                        "is_weighted_aic_winner": int(fit["model"] == winner),
                    }
                )
    return pd.DataFrame(fit_rows), pd.DataFrame(prediction_rows), mass_by_panel


def _frontier_score_at(
    frontiers: pd.DataFrame, benchmark_ids: list[str], date: pd.Timestamp
) -> tuple[float, int]:
    values = []
    for benchmark_id in benchmark_ids:
        history = frontiers.loc[
            (frontiers["benchmark_id"] == benchmark_id)
            & (frontiers["date"] <= date),
            "frontier_score",
        ]
        if not history.empty:
            values.append(float(history.iloc[-1]))
    return (float(np.mean(values)) if values else np.nan, len(values))


def rolling_origin_backtest(
    obs: pd.DataFrame,
    frontiers: pd.DataFrame,
    panels: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run two explicitly different 90/180-day retrospective forecasts.

    ``fixed31_common_window`` begins only after all 31 fixed benchmarks have
    appeared and keeps those same 31 benchmarks at every origin.
    ``all44_delayed_entry`` admits each of the 44 observed benchmarks at its
    first observation and freezes the active subset separately at every
    origin.  Later entrants are never added to that origin's target outcome.
    """
    detail: list[dict[str, object]] = []
    first_dates = frontiers.groupby("benchmark_id")["date"].min().to_dict()
    cutoff = pd.Timestamp(frontiers["date"].max())
    configurations = (
        {
            "forecast_panel": "fixed31_common_window",
            "source_panel": "fixed31",
            "cohort_mode": "complete_fixed_panel",
        },
        {
            "forecast_panel": "all44_delayed_entry",
            "source_panel": "all44",
            "cohort_mode": "delayed_entry_active_at_origin",
        },
    )
    for configuration in configurations:
        forecast_panel = str(configuration["forecast_panel"])
        source_panel = str(configuration["source_panel"])
        cohort_mode = str(configuration["cohort_mode"])
        ids = panels[source_panel]
        if cohort_mode == "complete_fixed_panel":
            origin_start = max(first_dates[value] for value in ids)
        else:
            origin_start = min(first_dates[value] for value in ids) + pd.Timedelta(
                days=360
            )
        origins = pd.date_range(
            origin_start,
            cutoff - pd.Timedelta(days=90),
            freq="90D",
        )
        for origin in origins:
            if cohort_mode == "complete_fixed_panel":
                active = list(ids)
            else:
                active = [value for value in ids if first_dates[value] <= origin]
            if len(active) < 5:
                continue
            records, _ = build_mass_records(obs, active, cutoff=pd.Timestamp(origin))
            if records.empty:
                continue
            current, current_count = _frontier_score_at(
                frontiers, active, pd.Timestamp(origin)
            )
            fits = [fit_model(records, model_name) for model_name in MODEL_NAMES]
            for horizon in (90, 180):
                target = pd.Timestamp(origin) + pd.Timedelta(days=horizon)
                if target > cutoff:
                    continue
                actual, actual_count = _frontier_score_at(frontiers, active, target)
                if actual_count != current_count:
                    raise ValueError("The rolling-origin active panel changed unexpectedly")
                candidates: list[tuple[str, float]] = [("no_change", current)]
                for fit in fits:
                    pair = pd.Series([pd.Timestamp(origin), target])
                    origin_cdf, target_cdf = predict_model(fit, records, pair)
                    conditional_unlock = max(
                        0.0,
                        min(
                            1.0,
                            (target_cdf - origin_cdf)
                            / max(1.0 - origin_cdf, 1e-9),
                        ),
                    )
                    prediction = current + (100.0 - current) * conditional_unlock
                    candidates.append((str(fit["model"]), float(prediction)))
                for model, prediction in candidates:
                    remaining_score_mass = max(100.0 - current, 0.0)
                    if remaining_score_mass <= 1e-12:
                        conditional_prediction = 0.0
                        conditional_outcome = 0.0
                    else:
                        conditional_prediction = float(
                            np.clip(
                                (prediction - current) / remaining_score_mass,
                                0,
                                1,
                            )
                        )
                        conditional_outcome = float(
                            np.clip(
                                (actual - current) / remaining_score_mass,
                                0,
                                1,
                            )
                        )
                    detail.append(
                        {
                            "panel": forecast_panel,
                            "source_panel": source_panel,
                            "cohort_mode": cohort_mode,
                            "nominal_panel_benchmark_count": len(ids),
                            "origin_date": origin,
                            "target_date": target,
                            "horizon_days": horizon,
                            "model": model,
                            "active_benchmark_count": len(active),
                            "current_score": current,
                            "predicted_score": prediction,
                            "actual_score": actual,
                            "error": prediction - actual,
                            "absolute_error": abs(prediction - actual),
                            "squared_error": (prediction - actual) ** 2,
                            "remaining_score_mass_at_origin": remaining_score_mass,
                            "conditional_predicted_future_hit": (
                                conditional_prediction
                            ),
                            "conditional_actual_future_hit": conditional_outcome,
                            "brier": float(
                                conditional_outcome
                                * (1 - conditional_prediction) ** 2
                                + (1 - conditional_outcome)
                                * conditional_prediction**2
                            ),
                            "backtest_scope": BACKTEST_CAVEAT,
                        }
                    )
    detail_frame = pd.DataFrame(detail)
    if detail_frame.empty:
        return detail_frame, pd.DataFrame()
    summary = (
        detail_frame.groupby(
            [
                "panel",
                "source_panel",
                "cohort_mode",
                "nominal_panel_benchmark_count",
                "horizon_days",
                "model",
            ],
            as_index=False,
        )
        .agg(
            forecast_count=("error", "size"),
            min_active_benchmark_count=("active_benchmark_count", "min"),
            max_active_benchmark_count=("active_benchmark_count", "max"),
            mse=("squared_error", "mean"),
            mae=("absolute_error", "mean"),
            brier=("brier", "mean"),
        )
        .sort_values(["panel", "horizon_days", "brier", "mae"])
    )
    summary["rmse"] = np.sqrt(summary["mse"])
    baseline = summary.loc[
        summary["model"] == "no_change",
        ["panel", "horizon_days", "mse", "brier"],
    ].rename(
        columns={
            "mse": "no_change_mse",
            "brier": "no_change_brier",
        }
    )
    summary = summary.merge(
        baseline,
        on=["panel", "horizon_days"],
        how="left",
        validate="many_to_one",
    )
    summary["mse_skill"] = np.where(
        summary["no_change_mse"] > 0,
        1.0 - summary["mse"] / summary["no_change_mse"],
        np.nan,
    )
    summary["brier_skill"] = np.where(
        summary["no_change_brier"] > 0,
        1.0 - summary["brier"] / summary["no_change_brier"],
        np.nan,
    )
    summary.loc[summary["model"] == "no_change", ["mse_skill", "brier_skill"]] = 0.0
    summary["backtest_scope"] = BACKTEST_CAVEAT
    return detail_frame, summary


def cluster_bootstrap(
    mass_by_panel: dict[str, pd.DataFrame],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Benchmark-cluster bootstrap; duplicate clusters receive duplicate weight."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for panel in ("fixed31", "all44"):
        records = mass_by_panel[panel]
        ids = records["benchmark_id"].unique()
        for replicate in range(replicates):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            counts = pd.Series(sampled).value_counts()
            boot = records.loc[records["benchmark_id"].isin(counts.index)].copy()
            boot["fit_weight"] = (
                boot["mass"]
                * boot["benchmark_id"].map(counts).to_numpy(dtype=float)
            )
            for model_name in MODEL_NAMES:
                fit = fit_model(boot, model_name)
                clean = {
                    key: value
                    for key, value in fit.items()
                    if key not in {"_theta", "optimizer_message"}
                }
                rows.append(
                    {
                        "panel": panel,
                        "replicate": replicate,
                        "resampled_cluster_count": len(ids),
                        "unique_cluster_count": int(len(counts)),
                        **clean,
                    }
                )
    return pd.DataFrame(rows)


def jump_concentration_ablation(
    mass_by_panel: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Refit after removing the largest 1/3/5/10 exact frontier jumps."""
    rows: list[dict[str, object]] = []
    for panel in ("fixed31", "all44"):
        records = mass_by_panel[panel]
        exact = records.loc[
            records["record_type"] == "exact_tied_event"
        ].sort_values("mass", ascending=False)
        for drop_count in (0, 1, 3, 5, 10):
            dropped_indices = exact.head(drop_count).index
            ablated = records.drop(index=dropped_indices).copy()
            removed = records.loc[dropped_indices]
            dropped_labels = ";".join(
                f"{row.benchmark_id}@{row.event_date.date()}:{row.mass:.4f}"
                for row in removed.itertuples()
            )
            for model_name in MODEL_NAMES:
                fit = fit_model(ablated, model_name)
                clean = {
                    key: value
                    for key, value in fit.items()
                    if key not in {"_theta", "optimizer_message"}
                }
                rows.append(
                    {
                        "panel": panel,
                        "dropped_top_exact_jumps": drop_count,
                        "removed_exact_event_mass": float(removed["mass"].sum()),
                        "ablation_aic_warning": (
                            "AIC is comparable across models within this "
                            "ablation, not across drop counts because total "
                            "composite weight changes"
                        ),
                        "dropped_events": dropped_labels,
                        **clean,
                    }
                )
    return pd.DataFrame(rows)


def jump_time_permutation_null(
    mass_by_panel: dict[str, pd.DataFrame],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Permute exact-event dates against jump masses.

    The statistic is the mass-weighted mean calendar date of exact events.
    This null preserves the observed multiset of event dates and jump sizes but
    destroys their pairing.  It is a concentration diagnostic, not a no-
    progress null and not a test of latent learning.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for panel in ("fixed31", "all44"):
        exact = mass_by_panel[panel].loc[
            mass_by_panel[panel]["record_type"] == "exact_tied_event"
        ]
        dates = exact["time_years"].to_numpy(dtype=float)
        mass = exact["mass"].to_numpy(dtype=float)
        observed = float(np.average(dates, weights=mass))
        values = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            values[replicate] = np.average(rng.permutation(dates), weights=mass)
        distance = abs(observed - float(np.mean(values)))
        p_value = (1 + int(np.sum(np.abs(values - values.mean()) >= distance))) / (
            replicates + 1
        )
        rows.append(
            {
                "panel": panel,
                "null": "permute_exact_event_dates_against_jump_masses",
                "exact_event_count": len(exact),
                "replicates": replicates,
                "observed_mass_weighted_mean_event_date": _years_to_date(observed),
                "null_mean_date": _years_to_date(float(np.mean(values))),
                "null_p025_date": _years_to_date(float(np.quantile(values, 0.025))),
                "null_p975_date": _years_to_date(float(np.quantile(values, 0.975))),
                "two_sided_p_value": p_value,
                "interpretation": (
                    "tests whether large and small observed jumps are unusually "
                    "paired with early/late dates; not a latent-learning test"
                ),
            }
        )
    return pd.DataFrame(rows)


def _records_from_simulated_scores(
    schedules: dict[str, pd.DatetimeIndex],
    score_paths: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build mass records from simulated monotone scores on matched schedules."""
    rows: list[dict[str, object]] = []
    for benchmark_id, dates in schedules.items():
        scores = np.maximum.accumulate(np.asarray(score_paths[benchmark_id]) * 100)
        if len(dates) != len(scores):
            raise ValueError("Simulation schedule and score path length differ")
        first_score = float(scores[0])
        if first_score > 0:
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "record_type": "left_censored",
                    "event_date": dates[0],
                    "mass": first_score / 100,
                }
            )
        previous = first_score
        for date, score in zip(dates[1:], scores[1:]):
            delta = float(score - previous)
            if delta > 1e-12:
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "record_type": "exact_tied_event",
                        "event_date": date,
                        "mass": delta / 100,
                    }
                )
                previous = float(score)
        if previous < 100:
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "record_type": "right_censored",
                    "event_date": dates[-1],
                    "mass": 1 - previous / 100,
                }
            )
    records = pd.DataFrame(rows)
    records["event_date"] = pd.to_datetime(records["event_date"])
    records["time_years"] = _date_to_years(records["event_date"])
    records["fit_weight"] = records["mass"]
    return records


def matched_schedule_simulation(
    obs: pd.DataFrame,
    benchmark_ids: list[str],
    replicates: int,
    seed: int,
    atoms_per_benchmark: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate the exact-date analysis on actual evaluation schedules.

    ``clean_logistic`` shares one latent distribution across benchmarks.
    ``benchmark_heterogeneity`` draws benchmark-specific midpoints.
    ``heterogeneity_selective_stopping`` additionally truncates the later half
    of schedules for harder-than-median benchmarks, a deliberately informative
    observation process.
    """
    daily = _daily_best(obs, benchmark_ids)
    base_schedules = {
        benchmark_id: pd.DatetimeIndex(group["date"].sort_values())
        for benchmark_id, group in daily.groupby("benchmark_id")
    }
    grid_dates = pd.date_range(daily["date"].min(), daily["date"].max(), periods=100)
    grid_time = _date_to_years(grid_dates)
    all_time = _date_to_years(daily["date"])
    true_midpoint = float(np.median(all_time))
    true_scale = 0.65
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    scenarios = (
        "clean_logistic",
        "benchmark_heterogeneity",
        "heterogeneity_selective_stopping",
    )
    for scenario in scenarios:
        for replicate in range(replicates):
            schedules: dict[str, pd.DatetimeIndex] = {}
            score_paths: dict[str, np.ndarray] = {}
            midpoint_values: list[float] = []
            for benchmark_id, raw_dates in base_schedules.items():
                midpoint_shift = (
                    rng.normal(0, 0.8) if scenario != "clean_logistic" else 0.0
                )
                midpoint = true_midpoint + midpoint_shift
                midpoint_values.append(midpoint)
                uniform = np.clip(
                    rng.random(atoms_per_benchmark), 1e-8, 1 - 1e-8
                )
                unlock_times = midpoint + true_scale * np.log(
                    uniform / (1 - uniform)
                )
                dates = raw_dates
                if (
                    scenario == "heterogeneity_selective_stopping"
                    and midpoint_shift > 0
                    and len(dates) > 2
                ):
                    dates = dates[: max(2, math.ceil(len(dates) / 2))]
                times = _date_to_years(dates)
                scores = np.asarray(
                    [(unlock_times <= value).mean() for value in times], dtype=float
                )
                schedules[benchmark_id] = dates
                score_paths[benchmark_id] = scores
            records = _records_from_simulated_scores(schedules, score_paths)
            fit = fit_model(records, "logistic")
            predicted = predict_model(fit, records, pd.Series(grid_dates))
            midpoint_array = np.asarray(midpoint_values)
            truth = np.mean(
                expit(
                    (grid_time[:, None] - midpoint_array[None, :]) / true_scale
                ),
                axis=1,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "replicate": replicate,
                    "benchmark_count": len(schedules),
                    "atoms_per_benchmark": atoms_per_benchmark,
                    "fitted_midpoint_date": fit["midpoint_date"],
                    "fitted_midpoint_years_from_epoch": fit[
                        "midpoint_years_from_epoch"
                    ],
                    "fitted_scale_years": fit["scale_years"],
                    "true_common_midpoint_date": _years_to_date(true_midpoint),
                    "true_scale_years": true_scale,
                    "mean_benchmark_midpoint_date": _years_to_date(
                        float(midpoint_array.mean())
                    ),
                    "midpoint_bias_years_vs_mean": float(
                        fit["midpoint_years_from_epoch"] - midpoint_array.mean()
                    ),
                    "scale_bias_years": float(fit["scale_years"] - true_scale),
                    "curve_rmse_probability": float(
                        np.sqrt(np.mean(np.square(predicted - truth)))
                    ),
                    "final_curve_bias_probability": float(predicted[-1] - truth[-1]),
                    "converged": fit["converged"],
                    "calibration_scope": (
                        "latent logistic threshold atoms observed only on each "
                        "benchmark's matched empirical evaluation schedule"
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("scenario", as_index=False)
        .agg(
            replicates=("replicate", "size"),
            median_midpoint_bias_years=("midpoint_bias_years_vs_mean", "median"),
            p025_midpoint_bias_years=(
                "midpoint_bias_years_vs_mean",
                lambda values: float(np.quantile(values, 0.025)),
            ),
            p975_midpoint_bias_years=(
                "midpoint_bias_years_vs_mean",
                lambda values: float(np.quantile(values, 0.975)),
            ),
            median_scale_bias_years=("scale_bias_years", "median"),
            median_curve_rmse_probability=("curve_rmse_probability", "median"),
            p975_curve_rmse_probability=(
                "curve_rmse_probability",
                lambda values: float(np.quantile(values, 0.975)),
            ),
            median_final_curve_bias_probability=(
                "final_curve_bias_probability",
                "median",
            ),
            convergence_rate=("converged", "mean"),
        )
        .sort_values("scenario")
    )
    return detail, summary


def bootstrap_summary(bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Compact percentile summary of cluster-bootstrap model parameters."""
    if bootstrap.empty:
        return pd.DataFrame()
    rows = []
    for (panel, model), group in bootstrap.groupby(["panel", "model"]):
        for parameter in (
            "midpoint_years_from_epoch",
            "scale_years",
            "rate_per_year",
            "unlockable_fraction",
            "weighted_aic",
        ):
            values = pd.to_numeric(group[parameter], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "panel": panel,
                    "model": model,
                    "parameter": parameter,
                    "replicates": len(values),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p025": float(values.quantile(0.025)),
                    "p975": float(values.quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV output directory (default: project output/)",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=60,
        help="Benchmark-cluster bootstrap replicates per main panel",
    )
    parser.add_argument(
        "--permutation-reps",
        type=int,
        default=2000,
        help="Jump-time permutation replicates per main panel",
    )
    parser.add_argument(
        "--simulation-reps",
        type=int,
        default=30,
        help="Matched-schedule simulation replicates per scenario",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.bootstrap_reps, args.permutation_reps, args.simulation_reps
    ) < 1:
        raise ValueError("All replicate counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta, obs = load_data()
    frontiers = make_frontiers(obs)
    panels, membership = define_panels(meta, frontiers)

    all_records, all_intervals = build_mass_records(obs, panels["all44"])
    membership.to_csv(args.output_dir / "unlock_panel_definitions.csv", index=False)
    all_records.drop(columns=["time_years", "fit_weight"]).to_csv(
        args.output_dir / "unlock_mass_records.csv", index=False
    )
    all_intervals.to_csv(
        args.output_dir / "unlock_interval_diagnostics.csv", index=False
    )

    fits, predictions, mass_by_panel = fit_panels(obs, panels)
    fits.to_csv(args.output_dir / "unlock_model_fits.csv", index=False)
    predictions.to_csv(
        args.output_dir / "unlock_model_predictions.csv", index=False
    )

    forecast_detail, forecast_summary = rolling_origin_backtest(
        obs, frontiers, panels
    )
    forecast_detail.to_csv(
        args.output_dir / "unlock_forecast_detail.csv", index=False
    )
    forecast_summary.to_csv(
        args.output_dir / "unlock_forecast_summary.csv", index=False
    )

    bootstrap = cluster_bootstrap(
        mass_by_panel, replicates=args.bootstrap_reps, seed=args.seed + 10
    )
    bootstrap.to_csv(args.output_dir / "unlock_bootstrap.csv", index=False)
    bootstrap_summary(bootstrap).to_csv(
        args.output_dir / "unlock_bootstrap_summary.csv", index=False
    )

    ablation = jump_concentration_ablation(mass_by_panel)
    ablation.to_csv(
        args.output_dir / "unlock_jump_ablation.csv", index=False
    )
    permutation = jump_time_permutation_null(
        mass_by_panel,
        replicates=args.permutation_reps,
        seed=args.seed + 20,
    )
    permutation.to_csv(
        args.output_dir / "unlock_permutation_null.csv", index=False
    )

    simulation, simulation_summary = matched_schedule_simulation(
        obs,
        panels["fixed31"],
        replicates=args.simulation_reps,
        seed=args.seed + 30,
    )
    simulation.to_csv(
        args.output_dir / "unlock_simulation_calibration.csv", index=False
    )
    simulation_summary.to_csv(
        args.output_dir / "unlock_simulation_summary.csv", index=False
    )

    manifest = pd.DataFrame(
        [
            {
                "seed": args.seed,
                "benchmark_observations": len(obs),
                "benchmarks_with_eligible_frontiers": len(panels["all44"]),
                "fixed_panel_benchmarks": len(panels["fixed31"]),
                "mass_record_count": len(all_records),
                "mass_total": float(all_records["mass"].sum()),
                "bootstrap_replicates_per_main_panel": args.bootstrap_reps,
                "permutation_replicates_per_main_panel": args.permutation_reps,
                "simulation_replicates_per_scenario": args.simulation_reps,
                "time_coordinate": "real calendar years since 2018-01-01",
                "date_estimand": (
                    "retrospective capability-date observed first-hit record; "
                    "not always public-evidence date"
                ),
                "likelihood_warning": (
                    "weighted composite likelihood; threshold atoms are not iid"
                ),
                "interval_warning": INTERVAL_CAVEAT,
                "backtest_warning": BACKTEST_CAVEAT,
                "panel_counts_json": json.dumps(
                    {key: len(value) for key, value in panels.items()},
                    sort_keys=True,
                ),
            }
        ]
    )
    manifest.to_csv(args.output_dir / "unlock_run_manifest.csv", index=False)

    winners = fits.loc[fits["aic_winner"] == 1, ["panel", "model", "weighted_aic"]]
    print(
        f"Wrote unlock analysis for {len(panels['all44'])} benchmarks "
        f"({len(panels['fixed31'])} fixed-panel)."
    )
    print("Weighted-AIC winners:")
    print(winners.to_string(index=False))
    print("Simulation summary:")
    print(simulation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
