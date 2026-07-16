#!/usr/bin/env python3
"""Build EdgeBench-style benchmark frontiers and calendar-time fits.

Following EdgeBench, every benchmark is represented on its native 0--100 task
score scale and benchmark aggregates are equal-task arithmetic means.  Random
guessing floors are *not* subtracted.  The chance-adjusted value is retained as
a diagnostic column only and is never used for the frontier or aggregate.

The script intentionally distinguishes EdgeBench's within-run log-sigmoid law
from calendar-time progress.  Calendar time is tested with four simple
two-parameter models plus the three-parameter Edge calendar generalization on
the 0--100 task-score scale:

* linear: score ~ date
* log-score: log(score) ~ date (EdgeBench release-frontier analogue)
* log-error: log(100-score) ~ date
* logit: logit(score/100) ~ date
* Edge calendar: S_max/(1+exp[-k(date-date_mid)]) with fitted S_max
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/model-improvement-score-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
MIN_FRONTIER_POINTS_TO_PLOT = 3


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(DATA / "benchmark_metadata.csv")
    obs = pd.read_csv(DATA / "benchmark_observations.csv")
    obs["date"] = pd.to_datetime(obs["date"])
    obs["score"] = pd.to_numeric(obs["score"])
    obs["frontier_eligible"] = pd.to_numeric(obs["frontier_eligible"]).astype(int)
    obs = obs.merge(
        meta[
            [
                "benchmark_id",
                "benchmark_name",
                "category",
                "domain",
                "score_floor",
                "score_ceiling",
                "composite_group",
                "include_in_composite",
            ]
        ],
        on="benchmark_id",
        how="left",
        validate="many_to_one",
    )
    if obs["benchmark_name"].isna().any():
        missing = obs.loc[obs["benchmark_name"].isna(), "benchmark_id"].unique()
        raise ValueError(f"Missing benchmark metadata: {missing}")
    # EdgeBench first rescales each task run to its 0--100 task scale and then
    # averages tasks.  The benchmarks in this dataset already publish scores
    # on that scale, so their native score is the Edge score.  Do not subtract
    # chance: EdgeBench's fitted S_max handles attainable support at fit time.
    obs["edge_score"] = obs["score"].clip(0, 100)

    # Kept only to make the alternative normalization auditable.  No frontier,
    # composite, fit, or plot below uses this diagnostic column.
    span = obs["score_ceiling"] - obs["score_floor"]
    obs["chance_adjusted_score"] = (
        100.0 * (obs["score"] - obs["score_floor"]) / span
    ).clip(0, 100)
    return meta, obs


def make_frontiers(obs: pd.DataFrame) -> pd.DataFrame:
    eligible_obs = obs.loc[obs["frontier_eligible"] == 1].copy()
    rows = []
    for bench_id, g in eligible_obs.groupby("benchmark_id", sort=False):
        # One public-evidence date can contain several models. Keep the best.
        idx = g.groupby("date")["edge_score"].idxmax()
        daily = g.loc[idx].sort_values("date").copy()
        daily["frontier_score"] = daily["edge_score"].cummax()
        daily["is_frontier_improvement"] = daily["frontier_score"].gt(
            daily["frontier_score"].shift(fill_value=-np.inf)
        )
        daily = daily.loc[daily["is_frontier_improvement"]].copy()
        daily["delta_unlocked"] = daily["frontier_score"].diff()
        daily["left_censored"] = False
        if not daily.empty:
            first_idx = daily.index[0]
            daily.loc[first_idx, "delta_unlocked"] = daily.loc[first_idx, "frontier_score"]
            daily.loc[first_idx, "left_censored"] = True
        rows.append(daily)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["benchmark_id", "date"]).reset_index(drop=True)


def composite_groups(meta: pd.DataFrame) -> dict[str, list[str]]:
    eligible = meta.loc[meta["include_in_composite"] == 1]
    return {
        "All-benchmark frontier": eligible["benchmark_id"].tolist(),
    }


def _time_years(dates: pd.Series) -> np.ndarray:
    return (dates - dates.min()).dt.total_seconds().to_numpy() / (365.25 * 86400)


def _r2_rmse(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = actual - predicted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float("nan") if sst <= 1e-12 else 1.0 - sse / sst
    rmse = math.sqrt(sse / len(actual))
    return r2, rmse


def _aicc(actual: np.ndarray, predicted: np.ndarray, n_params: int) -> float:
    """Small-sample Akaike criterion on raw-score residuals."""
    n = len(actual)
    sse = float(np.sum((actual - predicted) ** 2))
    if n <= n_params + 1 or sse <= 0:
        return float("nan")
    aic = n * math.log(sse / n) + 2 * n_params
    return aic + 2 * n_params * (n_params + 1) / (n - n_params - 1)


def fit_calendar_models(dates: pd.Series, scores: pd.Series) -> dict[str, dict]:
    t = _time_years(dates)
    y = scores.to_numpy(dtype=float)
    eps = 0.25
    fits: dict[str, dict] = {}

    def store(name: str, transformed: np.ndarray, inverse) -> None:
        slope, intercept = np.polyfit(t, transformed, 1)
        predicted = np.asarray(inverse(intercept + slope * t), dtype=float)
        predicted = np.clip(predicted, 0, 100)
        r2, rmse = _r2_rmse(y, predicted)
        fits[name] = {
            "slope_per_year": float(slope),
            "intercept": float(intercept),
            "r2": r2,
            "rmse": rmse,
            "aicc": _aicc(y, predicted, 2),
            "predicted": predicted,
        }

    store("linear", y, lambda z: z)
    store("log_score", np.log(np.clip(y, eps, None)), np.exp)
    store("log_error", np.log(np.clip(100.0 - y, eps, None)), lambda z: 100 - np.exp(z))
    p = np.clip(y / 100.0, eps / 100.0, 1 - eps / 100.0)
    store("logit", np.log(p / (1 - p)), lambda z: 100 / (1 + np.exp(-z)))
    return fits


def _calendar_logistic(t: np.ndarray, s_max: float, t_mid: float, k: float) -> np.ndarray:
    """EdgeBench's logistic frontier in a calendar-time learning coordinate."""
    z = np.clip(k * (t - t_mid), -60, 60)
    return s_max / (1.0 + np.exp(-z))


def fit_edge_calendar(dates: pd.Series, scores: pd.Series) -> dict[str, float | np.ndarray]:
    """Fit S(d)=S_max/(1+exp[-k(d-d_mid)]) on the raw 0--100 scale.

    EdgeBench uses log interaction time as its effective learning coordinate.
    For model-development history there is no natural calendar-time zero, so
    the main generalization uses calendar date itself as that coordinate.  The
    fitted-S_max log-odds view is then linear in calendar time with slope k.
    """
    t = _time_years(dates)
    y = scores.to_numpy(dtype=float)
    span = max(float(np.ptp(t)), 1.0 / 365.25)
    y_max = float(np.max(y))
    lower_s = max(y_max, 1e-6)
    upper_s = max(100.000001, lower_s + 1e-6)
    p_s = min(max(y_max + max(1.0, 0.1 * (100.0 - y_max)), lower_s + 1e-7), upper_s - 1e-7)
    p_mid = float(np.median(t))
    p_k = max(0.05, min(5.0, 4.0 / span))
    try:
        params, _ = curve_fit(
            _calendar_logistic,
            t,
            y,
            p0=[p_s, p_mid, p_k],
            bounds=(
                [lower_s, float(np.min(t) - 12 * span), 1e-5],
                [upper_s, float(np.max(t) + 12 * span), 100.0],
            ),
            maxfev=100000,
        )
        s_max, t_mid, k = [float(v) for v in params]
        predicted = _calendar_logistic(t, s_max, t_mid, k)
        r2, rmse = _r2_rmse(y, predicted)
        eps = 1e-7
        observed_log_odds = np.log(
            np.clip(y, eps, None) / np.clip(s_max - y, eps, None)
        )
        fitted_log_odds = k * (t - t_mid)
        log_odds_r2, log_odds_rmse = _r2_rmse(observed_log_odds, fitted_log_odds)
        return {
            "s_max": s_max,
            "t_mid_years": t_mid,
            "k_per_year": k,
            "r2": r2,
            "rmse": rmse,
            "aicc": _aicc(y, predicted, 3),
            "log_odds_r2": log_odds_r2,
            "log_odds_rmse": log_odds_rmse,
            "predicted": predicted,
            "observed_log_odds": observed_log_odds,
            "fitted_log_odds": fitted_log_odds,
            "s_max_boundary_gap": s_max - y_max,
        }
    except (RuntimeError, ValueError, FloatingPointError):
        return {
            "s_max": float("nan"),
            "t_mid_years": float("nan"),
            "k_per_year": float("nan"),
            "r2": float("nan"),
            "rmse": float("nan"),
            "aicc": float("nan"),
            "log_odds_r2": float("nan"),
            "log_odds_rmse": float("nan"),
            "predicted": np.full_like(y, np.nan),
            "observed_log_odds": np.full_like(y, np.nan),
            "fitted_log_odds": np.full_like(y, np.nan),
            "s_max_boundary_gap": float("nan"),
        }


def fit_summaries(frontiers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bench_id, g in frontiers.groupby("benchmark_id", sort=False):
        g = g.sort_values("date")
        base = {
            "benchmark_id": bench_id,
            "benchmark_name": g["benchmark_name"].iloc[0],
            "category": g["category"].iloc[0],
            "n_frontier_points": len(g),
            "start_date": g["date"].min().date().isoformat(),
            "end_date": g["date"].max().date().isoformat(),
            "start_score": g["frontier_score"].iloc[0],
            "end_score": g["frontier_score"].iloc[-1],
        }
        if len(g) < 3 or g["date"].nunique() < 3:
            base["best_fit"] = "insufficient_points"
            rows.append(base)
            continue
        fits = fit_calendar_models(g["date"], g["frontier_score"])
        for name, result in fits.items():
            base[f"{name}_r2"] = result["r2"]
            base[f"{name}_rmse"] = result["rmse"]
            base[f"{name}_slope_per_year"] = result["slope_per_year"]
        valid = {k: v for k, v in fits.items() if np.isfinite(v["r2"])}
        base["best_fit"] = max(valid, key=lambda k: valid[k]["r2"])
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["category", "benchmark_id"])


def composite_events(
    frontiers: pd.DataFrame, benchmark_ids: list[str], composite_name: str
) -> pd.DataFrame:
    subset = frontiers.loc[frontiers["benchmark_id"].isin(benchmark_ids)].copy()
    first_dates = subset.groupby("benchmark_id")["date"].min()
    missing = sorted(set(benchmark_ids) - set(first_dates.index))
    if missing:
        raise ValueError(f"No eligible frontier for {missing}")
    start = first_dates.max()
    event_dates = sorted(subset.loc[subset["date"] >= start, "date"].unique())
    rows = []
    previous = None
    for date in event_dates:
        values = []
        for bench_id in benchmark_ids:
            history = subset.loc[
                (subset["benchmark_id"] == bench_id) & (subset["date"] <= date)
            ]
            if history.empty:
                break
            values.append(float(history["frontier_score"].iloc[-1]))
        if len(values) != len(benchmark_ids):
            continue
        score = float(np.mean(values))
        if previous is None or abs(score - previous) > 1e-10:
            rows.append(
                {
                    "composite": composite_name,
                    "date": pd.Timestamp(date),
                    "score": score,
                    "benchmark_count": len(benchmark_ids),
                }
            )
            previous = score
    return pd.DataFrame(rows)


def build_composites(meta: pd.DataFrame, frontiers: pd.DataFrame) -> pd.DataFrame:
    groups = composite_groups(meta)
    parts = [composite_events(frontiers, ids, name) for name, ids in groups.items()]
    return pd.concat(parts, ignore_index=True)


def sample_composites_monthly(composites: pd.DataFrame) -> pd.DataFrame:
    """Sample each stepwise composite on a regular calendar grid.

    EdgeBench averages curves point by point on a common dense time grid.  A
    monthly grid is the calendar-time analogue and prevents release-heavy
    months from receiving more fit weight merely because they contain more
    announcements.
    """
    rows = []
    for name, g in composites.groupby("composite", sort=False):
        g = g.sort_values("date")
        start, end = g["date"].min(), g["date"].max()
        dates = sorted(
            set([start, end])
            | set(pd.date_range(start=start, end=end, freq="MS").to_pydatetime())
        )
        for date in dates:
            date = pd.Timestamp(date)
            history = g.loc[g["date"] <= date]
            if history.empty:
                continue
            rows.append(
                {
                    "composite": name,
                    "date": date,
                    "score": float(history["score"].iloc[-1]),
                    "benchmark_count": int(g["benchmark_count"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def composite_unlock_events(composites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in composites.groupby("composite", sort=False):
        g = g.sort_values("date").copy()
        g["delta_unlocked"] = g["score"].diff()
        g["left_censored"] = False
        g.loc[g.index[0], "delta_unlocked"] = g["score"].iloc[0]
        g.loc[g.index[0], "left_censored"] = True
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def composite_summaries(
    composites: pd.DataFrame, sampled_composites: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for name, g in composites.groupby("composite", sort=False):
        g = g.sort_values("date")
        row = {
            "composite": name,
            "benchmark_count": int(g["benchmark_count"].iloc[0]),
            "n_event_points": len(g),
            "start_date": g["date"].min().date().isoformat(),
            "end_date": g["date"].max().date().isoformat(),
            "start_score": g["score"].iloc[0],
            "end_score": g["score"].iloc[-1],
        }
        fit_g = sampled_composites.loc[
            sampled_composites["composite"] == name
        ].sort_values("date")
        row["n_monthly_fit_points"] = len(fit_g)
        if len(fit_g) >= 4:
            fits = fit_calendar_models(fit_g["date"], fit_g["score"])
            for fit_name, result in fits.items():
                row[f"{fit_name}_r2"] = result["r2"]
                row[f"{fit_name}_rmse"] = result["rmse"]
                row[f"{fit_name}_aicc"] = result["aicc"]
                row[f"{fit_name}_slope_per_year"] = result["slope_per_year"]
            edge_fit = fit_edge_calendar(fit_g["date"], fit_g["score"])
            row["edge_calendar_r2"] = edge_fit["r2"]
            row["edge_calendar_rmse"] = edge_fit["rmse"]
            row["edge_calendar_aicc"] = edge_fit["aicc"]
            row["edge_calendar_s_max"] = edge_fit["s_max"]
            row["edge_calendar_t_mid_years_from_panel_start"] = edge_fit["t_mid_years"]
            row["edge_calendar_k_per_year"] = edge_fit["k_per_year"]
            row["edge_calendar_log_odds_r2"] = edge_fit["log_odds_r2"]
            row["edge_calendar_s_max_boundary_gap"] = edge_fit["s_max_boundary_gap"]
            valid = {
                **{k: v for k, v in fits.items() if np.isfinite(v["r2"])},
                "edge_calendar": edge_fit,
            }
            best = max(valid, key=lambda k: valid[k]["r2"])
            row["best_fit"] = best
            valid_aicc = {
                k: v for k, v in valid.items() if np.isfinite(v.get("aicc", np.nan))
            }
            if valid_aicc:
                row["best_fit_by_aicc"] = min(
                    valid_aicc, key=lambda k: valid_aicc[k]["aicc"]
                )
            if best == "edge_calendar":
                slope = edge_fit["k_per_year"]
                if slope > 0:
                    row["characteristic_time_days"] = 365.25 * math.log(2) / slope
                    row["characteristic_time_label"] = "fitted frontier odds doubling"
                rows.append(row)
                continue
            slope = fits[best]["slope_per_year"]
            if best == "log_score" and slope > 0:
                row["characteristic_time_days"] = 365.25 * math.log(2) / slope
                row["characteristic_time_label"] = "score doubling"
            elif best == "log_error" and slope < 0:
                row["characteristic_time_days"] = -365.25 * math.log(2) / slope
                row["characteristic_time_label"] = "error-gap halving"
            elif best == "logit" and slope > 0:
                row["characteristic_time_days"] = 365.25 * math.log(2) / slope
                row["characteristic_time_label"] = "odds doubling"
        rows.append(row)
    return pd.DataFrame(rows)


def latest_snapshot(frontiers: pd.DataFrame) -> pd.DataFrame:
    idx = frontiers.groupby("benchmark_id")["date"].idxmax()
    cols = [
        "benchmark_id",
        "benchmark_name",
        "category",
        "domain",
        "date",
        "system",
        "score",
        "edge_score",
        "chance_adjusted_score",
        "frontier_score",
        "protocol",
        "source_url",
    ]
    return frontiers.loc[idx, cols].sort_values(["category", "domain", "benchmark_id"])


def plot_small_multiples(obs: pd.DataFrame, frontiers: pd.DataFrame, meta: pd.DataFrame) -> None:
    frontier_counts = frontiers.groupby("benchmark_id").size()
    order = [
        benchmark_id
        for benchmark_id in meta["benchmark_id"]
        if frontier_counts.get(benchmark_id, 0) >= MIN_FRONTIER_POINTS_TO_PLOT
    ]
    ncols = 4
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.25 * nrows), squeeze=False)
    for ax, bench_id in zip(axes.ravel(), order):
        g_all = obs.loc[
            (obs["benchmark_id"] == bench_id) & (obs["frontier_eligible"] == 1)
        ].sort_values("date")
        g_front = frontiers.loc[frontiers["benchmark_id"] == bench_id].sort_values("date")
        ax.scatter(g_all["date"], g_all["edge_score"], s=22, alpha=0.45, color="#5d6b7a")
        ax.step(
            g_front["date"],
            g_front["frontier_score"],
            where="post",
            linewidth=2,
            color="#1f77b4",
        )
        ax.scatter(g_front["date"], g_front["frontier_score"], s=24, color="#1f77b4")
        title = meta.loc[meta["benchmark_id"] == bench_id, "benchmark_name"].iloc[0]
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylim(-4, 104)
        ax.grid(axis="y", alpha=0.2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes.ravel()[len(order) :]:
        ax.axis("off")
    fig.suptitle(
        "Benchmark-wise public frontier progress "
        f"(≥{MIN_FRONTIER_POINTS_TO_PLOT} frontier events; Edge task scale 0–100)",
        fontsize=16,
        y=0.995,
    )
    fig.text(0.005, 0.5, "Native benchmark score (0–100)", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.015, 0.01, 1, 0.982])
    fig.savefig(OUTPUT / "benchmark_small_multiples.png", dpi=170)
    plt.close(fig)


def plot_composites(
    composites: pd.DataFrame, sampled_composites: pd.DataFrame, summary: pd.DataFrame
) -> None:
    name = summary["composite"].iloc[0]
    g = composites.loc[composites["composite"] == name].sort_values("date")
    fit_g = sampled_composites.loc[
        sampled_composites["composite"] == name
    ].sort_values("date")
    edge_fit = fit_edge_calendar(fit_g["date"], fit_g["score"])
    dense_dates = pd.Series(
        pd.date_range(fit_g["date"].min(), fit_g["date"].max(), periods=320)
    )
    dense_t = (
        (dense_dates - fit_g["date"].min()).dt.total_seconds().to_numpy()
        / (365.25 * 86400)
    )
    dense_pred = _calendar_logistic(
        dense_t,
        float(edge_fit["s_max"]),
        float(edge_fit["t_mid_years"]),
        float(edge_fit["k_per_year"]),
    )

    frontier_color = "#374151"
    fit_color = "#238b57"
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.plot(
        g["date"],
        g["score"],
        marker="o",
        markersize=5,
        linewidth=1.8,
        color=frontier_color,
        zorder=3,
    )
    ax.plot(dense_dates, dense_pred, linewidth=3, color=fit_color, zorder=4)
    all_scores = np.concatenate([g["score"].to_numpy(dtype=float), dense_pred])
    span = max(float(np.ptp(all_scores)), 1.0)
    ax.set_ylim(all_scores.min() - 0.14 * span, all_scores.max() + 0.18 * span)
    ax.annotate(
        "Observed all-benchmark frontier",
        xy=(g["date"].iloc[-1], g["score"].iloc[-1]),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        color=frontier_color,
        fontsize=9,
    )
    fit_label_index = int(0.63 * (len(dense_dates) - 1))
    ax.annotate(
        "Edge calendar fit",
        xy=(dense_dates.iloc[fit_label_index], dense_pred[fit_label_index]),
        xytext=(8, -18),
        textcoords="offset points",
        color=fit_color,
        fontsize=9,
    )
    ax.set_title("All-benchmark frontier on real calendar time", fontsize=14)
    ax.set_ylabel("Equal-benchmark score (0–100)")
    ax.set_xlabel("Calendar date")
    ax.grid(alpha=0.18)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.savefig(OUTPUT / "composite_indices.png", dpi=180)
    plt.close(fig)


def plot_fit_comparison(summary: pd.DataFrame) -> None:
    fit_names = ["linear", "log_score", "log_error", "logit", "edge_calendar"]
    labels = ["Linear", "Log score", "Log remaining error", "Fixed-100 logit", "Edge calendar"]
    values = np.array(
        [float(summary[f"{fit_name}_r2"].iloc[0]) for fit_name in fit_names]
    )
    aicc_winner = str(summary["best_fit_by_aicc"].iloc[0])
    display_labels = [
        f"{label}  ·  AICc winner" if fit_name == aicc_winner else label
        for fit_name, label in zip(fit_names, labels)
    ]
    y = np.arange(len(fit_names))
    xmin = max(0.0, float(values.min() - 0.012))
    colors = [
        "#d97706" if fit_name == aicc_winner else "#238b57" if fit_name == "edge_calendar" else "#64748b"
        for fit_name in fit_names
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.8), constrained_layout=True)
    ax.hlines(y, xmin, values, color="#cbd5e1", linewidth=2, zorder=1)
    ax.scatter(values, y, color=colors, s=95, zorder=3)
    for yi, value, color in zip(y, values, colors):
        ax.annotate(
            f"{value:.4f}",
            xy=(value, yi),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=10,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels)
    ax.invert_yaxis()
    ax.set_xlim(xmin, min(1.001, float(values.max() + 0.014)))
    ax.set_xlabel("R² on original 0–100 score (zoomed; higher is better)")
    ax.set_title("Calendar-time fit comparison · All-benchmark frontier", fontsize=14)
    ax.grid(axis="x", alpha=0.18)
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUTPUT / "composite_fit_comparison.png", dpi=180)
    plt.close(fig)


def plot_edge_calendar_fit(
    sampled_composites: pd.DataFrame, summary: pd.DataFrame
) -> None:
    """Show the fitted Edge calendar sigmoid and its log-odds linearization."""
    name = summary["composite"].iloc[0]
    g = sampled_composites.loc[
        sampled_composites["composite"] == name
    ].sort_values("date")
    edge_fit = fit_edge_calendar(g["date"], g["score"])
    s_max = float(edge_fit["s_max"])
    k = float(edge_fit["k_per_year"])
    raw_r2 = float(edge_fit["r2"])
    log_odds_r2 = float(edge_fit["log_odds_r2"])
    fit_color = "#238b57"
    observation_color = "#475569"

    dense_dates = pd.Series(
        pd.date_range(g["date"].min(), g["date"].max(), periods=320)
    )
    dense_t = (
        (dense_dates - g["date"].min()).dt.total_seconds().to_numpy()
        / (365.25 * 86400)
    )
    dense_pred = _calendar_logistic(
        dense_t,
        s_max,
        float(edge_fit["t_mid_years"]),
        k,
    )

    fig, (score_ax, odds_ax) = plt.subplots(
        2,
        1,
        figsize=(10.5, 9.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )
    score_ax.plot(
        g["date"],
        g["score"],
        color=observation_color,
        marker="o",
        markersize=5,
        linewidth=1.6,
        zorder=3,
    )
    score_ax.plot(dense_dates, dense_pred, color=fit_color, linewidth=3, zorder=4)
    score_ax.axhline(
        s_max,
        color=fit_color,
        linestyle=":",
        linewidth=1.8,
        alpha=0.8,
    )
    score_values = np.concatenate(
        [g["score"].to_numpy(dtype=float), dense_pred, np.array([s_max])]
    )
    score_span = max(float(np.ptp(score_values)), 1.0)
    score_ax.set_ylim(
        max(0.0, float(score_values.min() - 0.10 * score_span)),
        min(103.0, float(score_values.max() + 0.10 * score_span)),
    )
    score_ax.text(
        g["date"].max(),
        s_max,
        rf" fitted ceiling  $S_{{\max}}={s_max:.2f}$ ",
        ha="right",
        va="bottom",
        color=fit_color,
        fontsize=9,
    )
    observed_index = min(3, len(g) - 1)
    score_ax.annotate(
        "Monthly fixed-panel score",
        xy=(g["date"].iloc[observed_index], g["score"].iloc[observed_index]),
        xytext=(-4, 16),
        textcoords="offset points",
        color=observation_color,
        fontsize=9,
    )
    fit_label_index = int(0.67 * (len(dense_dates) - 1))
    score_ax.annotate(
        "Edge calendar fit",
        xy=(dense_dates.iloc[fit_label_index], dense_pred[fit_label_index]),
        xytext=(10, -18),
        textcoords="offset points",
        color=fit_color,
        fontsize=9,
    )
    score_ax.set_title(
        rf"Raw score · $R^2={raw_r2:.3f}$ · $k={k:.2f}$/yr",
        fontsize=11,
    )
    score_ax.set_ylabel("Equal-benchmark score (0–100)")
    score_ax.grid(alpha=0.18)

    observed_log_odds = np.asarray(edge_fit["observed_log_odds"], dtype=float)
    fitted_log_odds = np.asarray(edge_fit["fitted_log_odds"], dtype=float)
    odds_ax.scatter(
        g["date"],
        observed_log_odds,
        color=observation_color,
        s=36,
        zorder=3,
    )
    odds_ax.plot(g["date"], fitted_log_odds, color=fit_color, linewidth=3, zorder=4)
    line_label_index = int(0.58 * (len(g) - 1))
    odds_ax.annotate(
        "Linearized Edge calendar fit",
        xy=(g["date"].iloc[line_label_index], fitted_log_odds[line_label_index]),
        xytext=(10, -18),
        textcoords="offset points",
        color=fit_color,
        fontsize=9,
    )
    odds_ax.set_title(
        rf"EdgeBench linearization · fitted-ceiling log-odds $R^2={log_odds_r2:.3f}$",
        fontsize=11,
    )
    odds_ax.set_ylabel(r"$\log[S/(S_{\max}-S)]$")
    odds_ax.set_xlabel("Calendar date")
    odds_ax.grid(alpha=0.18)
    odds_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    odds_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("All-benchmark Edge calendar fit", fontsize=16)
    fig.savefig(OUTPUT / "edge_calendar_fit.png", dpi=180)
    plt.close(fig)


def plot_edge_linearization(
    sampled_composites: pd.DataFrame,
    summary: pd.DataFrame,
    frontiers: pd.DataFrame,
    meta: pd.DataFrame,
) -> None:
    """Plot the aggregate and genuine benchmark events with a reserved legend."""
    name = summary["composite"].iloc[0]
    groups = composite_groups(meta)
    benchmark_order = meta.loc[meta["include_in_composite"] == 1, "benchmark_id"].tolist()
    frontier_counts = frontiers.groupby("benchmark_id").size()
    benchmark_names = meta.set_index("benchmark_id")["benchmark_name"].to_dict()
    cmap = plt.get_cmap("turbo")
    benchmark_colors = {
        bench_id: cmap(i / max(len(benchmark_order) - 1, 1))
        for i, bench_id in enumerate(benchmark_order)
    }
    markers = ["o", "s", "^", "D", "v", "P"]
    fig = plt.figure(figsize=(13.5, 9.4), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[4.8, 1.65],
        height_ratios=[1.0, 1.35],
    )
    aggregate_ax = fig.add_subplot(grid[0, 0])
    events_ax = fig.add_subplot(grid[1, 0], sharex=aggregate_ax)
    legend_ax = fig.add_subplot(grid[:, 1])
    legend_ax.axis("off")
    eps = 0.25
    aggregate_color = "#111827"
    active_benchmark_ids: set[str] = set()
    g = sampled_composites.loc[
        sampled_composites["composite"] == name
    ].sort_values("date")
    panel_ids = groups[name]
    panel_event_count = 0

    for bench_id in panel_ids:
        if frontier_counts.get(bench_id, 0) < MIN_FRONTIER_POINTS_TO_PLOT:
            continue
        bench = frontiers.loc[
            (frontiers["benchmark_id"] == bench_id)
            & (frontiers["is_frontier_improvement"] == 1)
            & (frontiers["date"] >= g["date"].min())
            & (frontiers["date"] <= g["date"].max())
        ].sort_values("date")
        if bench.empty:
            continue
        panel_event_count += len(bench)
        active_benchmark_ids.add(bench_id)
        p = np.clip(
            bench["frontier_score"].to_numpy(dtype=float) / 100.0,
            eps / 100.0,
            1 - eps / 100.0,
        )
        events_ax.scatter(
            bench["date"],
            np.log(p / (1 - p)),
            color=benchmark_colors[bench_id],
            marker=markers[benchmark_order.index(bench_id) % len(markers)],
            s=30,
            alpha=0.62,
            zorder=2,
        )

    aggregate_p = np.clip(
        g["score"].to_numpy(dtype=float) / 100.0,
        eps / 100.0,
        1 - eps / 100.0,
    )
    aggregate_log_odds = np.log(aggregate_p / (1 - aggregate_p))
    logit_fit = fit_calendar_models(g["date"], g["score"])["logit"]
    t = _time_years(g["date"])
    fitted_log_odds = logit_fit["intercept"] + logit_fit["slope_per_year"] * t
    logit_space_r2, _ = _r2_rmse(aggregate_log_odds, fitted_log_odds)
    aggregate_ax.plot(
        g["date"],
        aggregate_log_odds,
        color=aggregate_color,
        marker="o",
        markersize=5.5,
        linewidth=2.8,
        zorder=5,
    )
    aggregate_ax.plot(
        g["date"],
        fitted_log_odds,
        color=aggregate_color,
        linestyle="--",
        linewidth=2.2,
        zorder=6,
    )
    aggregate_range = np.concatenate([aggregate_log_odds, fitted_log_odds])
    aggregate_span = float(np.ptp(aggregate_range))
    aggregate_pad = max(0.12 * aggregate_span, 0.06)
    aggregate_ax.set_ylim(
        float(aggregate_range.min() - aggregate_pad),
        float(aggregate_range.max() + aggregate_pad),
    )
    aggregate_ax.set_title(
        rf"Aggregate fixed-100 log-odds · $R^2={logit_space_r2:.3f}$",
        fontsize=12,
    )
    aggregate_ax.set_ylabel("Aggregate log-odds")
    aggregate_ax.grid(alpha=0.18)
    aggregate_ax.tick_params(axis="x", labelbottom=False)

    events_ax.set_title(
        f"{panel_event_count} genuine frontier events · "
        f"{len(active_benchmark_ids)}/{len(panel_ids)} benchmarks active in window",
        fontsize=11,
    )
    events_ax.set_ylabel(r"Benchmark events: $\log[S/(100-S)]$")
    events_ax.set_xlabel("Calendar date")
    events_ax.grid(alpha=0.18)
    events_ax.axhline(0, color="#6b7280", linewidth=0.8, alpha=0.6)
    events_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    events_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(
        "All-benchmark linearity and underlying frontier events",
        fontsize=16,
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=benchmark_colors[bench_id],
            marker=markers[i % len(markers)],
            linewidth=1.2,
            markersize=4,
            label=benchmark_names[bench_id],
        )
        for i, bench_id in enumerate(benchmark_order)
        if bench_id in active_benchmark_ids
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=aggregate_color,
                marker="o",
                linewidth=2.8,
                label="Aggregate observations",
            ),
            Line2D(
                [0],
                [0],
                color=aggregate_color,
                linestyle="--",
                linewidth=2.2,
                label="Aggregate logit fit",
            ),
        ]
    )
    legend_ax.set_title("Benchmark event key", loc="left", fontsize=12, pad=12)
    legend_ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        fontsize=8.5,
        borderaxespad=0,
        labelspacing=0.72,
        handlelength=1.8,
    )
    fig.savefig(OUTPUT / "edge_calendar_linearization.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    meta, obs = load_data()
    frontiers = make_frontiers(obs)
    bench_summary = fit_summaries(frontiers)
    composites = build_composites(meta, frontiers)
    sampled_composites = sample_composites_monthly(composites)
    comp_summary = composite_summaries(composites, sampled_composites)

    frontiers.to_csv(OUTPUT / "frontier_points.csv", index=False, date_format="%Y-%m-%d")
    frontiers[
        [
            "benchmark_id",
            "benchmark_name",
            "category",
            "date",
            "system",
            "frontier_score",
            "delta_unlocked",
            "left_censored",
            "source_url",
        ]
    ].to_csv(OUTPUT / "unlock_events.csv", index=False, date_format="%Y-%m-%d")
    bench_summary.to_csv(OUTPUT / "benchmark_fit_summary.csv", index=False)
    composites.to_csv(OUTPUT / "composite_points.csv", index=False, date_format="%Y-%m-%d")
    sampled_composites.to_csv(
        OUTPUT / "composite_monthly_points.csv", index=False, date_format="%Y-%m-%d"
    )
    composite_unlock_events(composites).to_csv(
        OUTPUT / "composite_unlock_events.csv", index=False, date_format="%Y-%m-%d"
    )
    comp_summary.to_csv(OUTPUT / "composite_fit_summary.csv", index=False)
    latest_snapshot(frontiers).to_csv(OUTPUT / "latest_snapshot.csv", index=False, date_format="%Y-%m-%d")

    if not args.no_plots:
        plot_small_multiples(obs, frontiers, meta)
        plot_composites(composites, sampled_composites, comp_summary)
        plot_fit_comparison(comp_summary)
        plot_edge_calendar_fit(sampled_composites, comp_summary)
        plot_edge_linearization(sampled_composites, comp_summary, frontiers, meta)

    print(f"observations={len(obs)} benchmarks={meta['benchmark_id'].nunique()}")
    print(comp_summary.to_string(index=False))


if __name__ == "__main__":
    main()
