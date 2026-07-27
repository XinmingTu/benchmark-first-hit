#!/usr/bin/env python3
"""Render the three compact evidence figures used by README.md and PAPER.md.

Run the two analysis scripts first:

    python scripts/analyze.py --no-plots
    python scripts/analyze_unlocks.py
    python scripts/plot_paper.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/model-improvement-score-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from analyze import load_data, make_frontiers
from analyze_unlocks import build_mass_records, define_panels


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"

INK = "#17202a"
MUTED = "#667085"
GRID = "#d8dee6"
PAPER = "#fbfaf7"
LEFT = "#718096"
EXACT = "#17805c"
RIGHT = "#d17a22"
LOGISTIC = "#17805c"
GOMPERTZ = "#6f5aa8"
PROBIT = "#3976a8"
EXPONENTIAL = "#c27736"
NO_CHANGE = "#7c8795"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#aab2bd",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(
        OUTPUT / filename,
        dpi=190,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def figure_measurement() -> None:
    meta, obs = load_data()
    frontiers = make_frontiers(obs)
    panels, _ = define_panels(meta, frontiers)
    fixed, _ = build_mass_records(obs, panels["fixed31"])
    all_records, _ = build_mass_records(obs, panels["all44"])

    fig = plt.figure(figsize=(15.2, 9.4))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.82, 1.75],
        width_ratios=[1.08, 1.25],
        hspace=0.34,
        wspace=0.38,
    )
    clock_ax = fig.add_subplot(grid[0, 0])
    mass_ax = fig.add_subplot(grid[0, 1])
    event_ax = fig.add_subplot(grid[1, :])

    # Panel A: the three clocks.
    clock_ax.set_axis_off()
    clock_ax.set_xlim(0, 1)
    clock_ax.set_ylim(0, 1)
    clock_ax.set_title("A  Three clocks, only one currently observed", loc="left")
    positions = [0.12, 0.50, 0.88]
    labels = [
        ("Latent unlock", r"$U_{bq}$", "not point-identified"),
        ("Corpus attribution", r"$R^{\mathcal{C}}_{bq}$", "current date axis"),
        ("Public disclosure", r"$T^{\mathcal{C}}_{bq}$", "needed for live forecasts"),
    ]
    colors = [MUTED, EXACT, PROBIT]
    for left, right in zip(positions[:-1], positions[1:]):
        clock_ax.add_patch(
            FancyArrowPatch(
                (left + 0.055, 0.56),
                (right - 0.055, 0.56),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.5,
                color="#9aa4b2",
            )
        )
    for x, (title, symbol, note), color in zip(positions, labels, colors):
        clock_ax.scatter(
            [x],
            [0.56],
            s=220,
            color=color,
            edgecolor=PAPER,
            linewidth=2,
            zorder=3,
        )
        clock_ax.text(x, 0.82, title, ha="center", va="center", fontweight="semibold")
        clock_ax.text(x, 0.56, symbol, ha="center", va="center", color="white", fontsize=12)
        clock_ax.text(x, 0.30, note, ha="center", va="center", color=MUTED, fontsize=9)
    clock_ax.text(
        0.5,
        0.08,
        r"Under complete evidence:  $U_{bq}\leq R^{\mathcal{C}}_{bq}\leq T^{\mathcal{C}}_{bq}$",
        ha="center",
        color=MUTED,
        fontsize=9,
    )

    # Panel B: how every benchmark's unit mass is partitioned.
    mass_ax.set_title("B  One unit of score mass per benchmark", loc="left")
    panels_for_bar = [
        ("Fixed panel · 31", fixed),
        ("All eligible · 44", all_records),
    ]
    record_types = ["left_censored", "exact_tied_event", "right_censored"]
    type_labels = ["Prevalent / left", "Dated jumps", "Unresolved / right"]
    type_colors = [LEFT, EXACT, RIGHT]
    y = np.arange(len(panels_for_bar))
    running = np.zeros(len(panels_for_bar))
    for record_type, label, color in zip(record_types, type_labels, type_colors):
        values = np.asarray(
            [
                100
                * records.loc[records["record_type"] == record_type, "mass"].sum()
                / records["benchmark_id"].nunique()
                for _, records in panels_for_bar
            ]
        )
        mass_ax.barh(
            y,
            values,
            left=running,
            height=0.48,
            color=color,
            edgecolor=PAPER,
            linewidth=1.4,
            label=label,
            zorder=2,
        )
        for row, (start, value) in enumerate(zip(running, values)):
            if value >= 10:
                mass_ax.text(
                    start + value / 2,
                    row,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="semibold",
                    fontsize=9,
                )
        running += values
    mass_ax.set_yticks(y)
    mass_ax.set_yticklabels([name for name, _ in panels_for_bar])
    mass_ax.set_xlim(0, 100)
    mass_ax.set_xlabel("Share of normalized score mass")
    mass_ax.invert_yaxis()
    mass_ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.47),
        fontsize=9,
        handlelength=1.4,
    )
    clean_axis(mass_ax, "x")

    # Panel C: each benchmark is one row; only genuine records are shown.
    first_dates = all_records.groupby("benchmark_id")["event_date"].min()
    category = all_records.groupby("benchmark_id")["category"].first()
    ordered_ids: list[str] = []
    group_bounds: list[tuple[str, int, int]] = []
    cursor = 0
    for group_name in ("model", "system", "agent"):
        ids = sorted(
            category.loc[category == group_name].index,
            key=lambda value: first_dates[value],
        )
        if not ids:
            continue
        ordered_ids.extend(ids)
        group_bounds.append((group_name, cursor, cursor + len(ids) - 1))
        cursor += len(ids)
    position = {benchmark_id: row for row, benchmark_id in enumerate(ordered_ids)}

    event_ax.set_title(
        "C  Evidence map · one row per benchmark, no carry-forward markers",
        loc="left",
    )
    for record_type, marker, color, edge, label in (
        ("left_censored", "<", LEFT, LEFT, "first observed mass"),
        ("exact_tied_event", "o", EXACT, EXACT, "genuine frontier jump"),
        ("right_censored", ">", PAPER, RIGHT, "last measured unresolved mass"),
    ):
        subset = all_records.loc[all_records["record_type"] == record_type].copy()
        sizes = 16 + 420 * subset["mass"].to_numpy()
        event_ax.scatter(
            subset["event_date"],
            subset["benchmark_id"].map(position),
            s=sizes,
            marker=marker,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.25,
            alpha=0.78,
            label=label,
            zorder=3,
        )
    for group_name, start, end in group_bounds:
        event_ax.axhspan(start - 0.5, end + 0.5, color="#9aa4b2", alpha=0.045)
        event_ax.text(
            -0.015,
            (start + end) / 2,
            group_name.upper(),
            transform=event_ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=MUTED,
            fontsize=9,
            fontweight="semibold",
        )
        if end < len(ordered_ids) - 1:
            event_ax.axhline(end + 0.5, color=GRID, linewidth=1)
    event_ax.set_ylim(-1, len(ordered_ids))
    event_ax.invert_yaxis()
    event_ax.set_yticks([])
    event_ax.set_ylabel("")
    event_ax.set_xlabel("Real calendar date")
    event_ax.xaxis.set_major_locator(mdates.YearLocator())
    event_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    clean_axis(event_ax, "x")
    event_ax.legend(
        ncol=3,
        loc="upper left",
        fontsize=9,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    event_ax.text(
        0.995,
        0.03,
        "Bubble area ∝ score mass",
        transform=event_ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=9,
    )

    fig.suptitle(
        "Benchmark First Hit measures an observed record process",
        x=0.065,
        ha="left",
        fontsize=18,
        fontweight="semibold",
        y=0.99,
    )
    fig.text(
        0.065,
        0.955,
        "First scores are prevalent mass; only later frontier jumps are dated events; stale remainder is measurement-censored.",
        ha="left",
        color=MUTED,
        fontsize=10.5,
    )
    save(fig, "first_hit_measurement.png")


def figure_forecasts() -> None:
    detail = pd.read_csv(
        OUTPUT / "unlock_forecast_detail.csv",
        parse_dates=["origin_date", "target_date"],
    )
    summary = pd.read_csv(OUTPUT / "unlock_forecast_summary.csv")
    delayed = detail.loc[detail["panel"] == "all44_delayed_entry"].copy()
    delayed["actual_gain"] = delayed["actual_score"] - delayed["current_score"]
    delayed["predicted_gain"] = delayed["predicted_score"] - delayed["current_score"]

    fig = plt.figure(figsize=(14.8, 9.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.36, wspace=0.42)
    model_style = {
        "logistic": (LOGISTIC, "o", "Logistic"),
        "gompertz": (GOMPERTZ, "^", "Gompertz"),
        "no_change": (NO_CHANGE, "s", "No change"),
    }
    for column, horizon in enumerate((90, 180)):
        ax = fig.add_subplot(grid[0, column])
        subset = delayed.loc[delayed["horizon_days"] == horizon]
        actual = (
            subset[["target_date", "actual_gain"]]
            .drop_duplicates()
            .sort_values("target_date")
        )
        ax.plot(
            actual["target_date"],
            actual["actual_gain"],
            color=INK,
            marker="D",
            markersize=4.8,
            linewidth=2.2,
            label="Realized gain",
            zorder=5,
        )
        for model, (color, marker, label) in model_style.items():
            group = subset.loc[subset["model"] == model].sort_values("target_date")
            ax.plot(
                group["target_date"],
                group["predicted_gain"],
                color=color,
                marker=marker,
                markersize=4.2,
                linewidth=1.65,
                linestyle="--" if model == "no_change" else "-",
                alpha=0.92,
                label=label,
                zorder=3,
            )
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_title(
            f"{chr(65 + column)}  {horizon}-day gain · delayed-entry cohort",
            loc="left",
        )
        ax.set_ylabel("Score points gained" if column == 0 else "")
        ax.set_xlabel("Forecast target date")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        clean_axis(ax, "y")
        if column == 0:
            ax.legend(ncol=2, loc="upper left", fontsize=8.7)

    # Panel C: all-eligible delayed-entry RMSE.
    rmse_ax = fig.add_subplot(grid[1, 0])
    show_models = ["logistic", "probit", "gompertz", "exponential_tail", "no_change"]
    labels = ["Logistic", "Probit", "Gompertz", "Exponential tail", "No change"]
    y = np.arange(len(show_models))
    for horizon, offset, marker, color in (
        (90, -0.11, "o", LOGISTIC),
        (180, 0.11, "s", PROBIT),
    ):
        values = (
            summary.loc[
                (summary["panel"] == "all44_delayed_entry")
                & (summary["horizon_days"] == horizon)
            ]
            .set_index("model")
            .loc[show_models, "rmse"]
            .to_numpy()
        )
        rmse_ax.scatter(
            values,
            y + offset,
            s=58,
            marker=marker,
            color=color,
            label=f"{horizon} days",
            zorder=4,
        )
        for value, yi in zip(values, y + offset):
            rmse_ax.text(value + 0.18, yi, f"{value:.1f}", va="center", fontsize=8, color=MUTED)
    rmse_ax.set_yticks(y)
    rmse_ax.set_yticklabels(labels)
    rmse_ax.invert_yaxis()
    rmse_ax.set_xlabel("RMSE on future score gain · points")
    rmse_ax.set_title("C  Error across 12 / 11 rolling origins", loc="left")
    rmse_ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.08),
        ncol=2,
        fontsize=8.7,
    )
    clean_axis(rmse_ax, "x")

    # Panel D: skill scores show uncertainty and the tiny common-window sample.
    skill_ax = fig.add_subplot(grid[1, 1])
    logistic = summary.loc[summary["model"] == "logistic"].copy()
    rows = [
        ("All eligible · 90d · n=12", "all44_delayed_entry", 90),
        ("All eligible · 180d · n=11", "all44_delayed_entry", 180),
        ("Fixed 31 · 90d · n=3", "fixed31_common_window", 90),
        ("Fixed 31 · 180d · n=2", "fixed31_common_window", 180),
    ]
    yy = np.arange(len(rows))
    mse_values = []
    brier_values = []
    for _, panel, horizon in rows:
        row = logistic.loc[
            (logistic["panel"] == panel)
            & (logistic["horizon_days"] == horizon)
        ].iloc[0]
        mse_values.append(float(row["mse_skill"]))
        brier_values.append(float(row["brier_skill"]))
    skill_ax.scatter(
        mse_values,
        yy - 0.10,
        marker="o",
        s=62,
        color=LOGISTIC,
        label="MSE skill",
        zorder=4,
    )
    skill_ax.scatter(
        brier_values,
        yy + 0.10,
        marker="s",
        s=56,
        color=PROBIT,
        label="Conditional Brier skill",
        zorder=4,
    )
    for mse, brier, yi in zip(mse_values, brier_values, yy):
        skill_ax.plot([brier, mse], [yi + 0.10, yi - 0.10], color=GRID, linewidth=1.2)
    skill_ax.axvline(0, color=NO_CHANGE, linewidth=1)
    skill_ax.set_yticks(yy)
    skill_ax.set_yticklabels([label for label, _, _ in rows])
    skill_ax.invert_yaxis()
    skill_ax.set_xlim(-0.04, 1.02)
    skill_ax.set_xlabel("Skill relative to no-change · higher is better")
    skill_ax.set_title("D  Logistic forecast skill", loc="left")
    skill_ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.08),
        ncol=2,
        fontsize=8.5,
    )
    clean_axis(skill_ax, "x")

    fig.suptitle(
        "The S-curve is tested by prediction, not visual fit",
        x=0.06,
        ha="left",
        fontsize=18,
        fontweight="semibold",
        y=0.99,
    )
    fig.text(
        0.06,
        0.955,
        "Retrospective pseudo-out-of-sample: capability dates and source discovery are not frozen historical vintages.",
        ha="left",
        color=MUTED,
        fontsize=10.5,
    )
    save(fig, "forecast_evidence.png")


def figure_robustness() -> None:
    fits = pd.read_csv(OUTPUT / "unlock_model_fits.csv")
    simulation = pd.read_csv(OUTPUT / "unlock_simulation_summary.csv")
    freshness = pd.read_csv(OUTPUT / "panel_freshness.csv")
    record_null = pd.read_csv(OUTPUT / "stationary_record_null.csv").iloc[0]

    fig = plt.figure(figsize=(14.8, 8.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.38, wspace=0.42)

    # Panel A: benchmark measurement age.
    fresh_ax = fig.add_subplot(grid[0, 0])
    ages = freshness["measurement_age_days"].to_numpy()
    counts = [
        int(np.sum(ages <= 90)),
        int(np.sum((ages > 90) & (ages <= 180))),
        int(np.sum((ages > 180) & (ages <= 365))),
        int(np.sum(ages > 365)),
    ]
    labels = ["≤90 days", "91–180", "181–365", ">365 days"]
    colors = [EXACT, PROBIT, RIGHT, "#a94b43"]
    bars = fresh_ax.barh(np.arange(4), counts, color=colors, height=0.55, zorder=3)
    for bar, count in zip(bars, counts):
        fresh_ax.text(
            count + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{count}",
            va="center",
            fontweight="semibold",
        )
    fresh_ax.set_yticks(np.arange(4))
    fresh_ax.set_yticklabels(labels)
    fresh_ax.invert_yaxis()
    fresh_ax.set_xlim(0, max(counts) + 4)
    fresh_ax.set_xlabel("Benchmarks in fixed panel")
    fresh_ax.set_title("A  Most remaining mass is weakly monitored", loc="left")
    clean_axis(fresh_ax, "x")

    # Panel B: link-function ambiguity across real-calendar vintages.
    shape_ax = fig.add_subplot(grid[0, 1])
    panels = [
        "fixed31",
        "all44",
        "vintage_pre2023",
        "vintage_2023",
        "vintage_2024",
        "vintage_2025plus",
    ]
    panel_labels = [
        "Fixed 31",
        "All 44",
        "Pre-2023",
        "2023",
        "2024",
        "2025+",
    ]
    models = ["logistic", "probit", "gompertz"]
    styles = {
        "logistic": (LOGISTIC, "o", "Logistic"),
        "probit": (PROBIT, "s", "Probit"),
        "gompertz": (GOMPERTZ, "^", "Gompertz"),
    }
    for model in models:
        values = []
        for panel in panels:
            group = fits.loc[fits["panel"] == panel]
            best = float(group["weighted_aic"].min())
            value = float(group.loc[group["model"] == model, "weighted_aic"].iloc[0])
            values.append(value - best)
        color, marker, label = styles[model]
        shape_ax.scatter(
            values,
            np.arange(len(panels)),
            s=55,
            marker=marker,
            color=color,
            label=label,
            zorder=4,
        )
    shape_ax.axvline(0, color=INK, linewidth=1)
    shape_ax.axvline(2, color=GRID, linewidth=1, linestyle="--")
    shape_ax.set_yticks(np.arange(len(panels)))
    shape_ax.set_yticklabels(panel_labels)
    shape_ax.invert_yaxis()
    shape_ax.set_xlabel("Δ weighted composite AIC from panel winner")
    shape_ax.set_title("B  No single link function wins every vintage", loc="left")
    shape_ax.legend(ncol=3, loc="lower right", fontsize=8.5)
    clean_axis(shape_ax, "x")

    # Panel C: the established stationary-record null from analyze.py.
    null_ax = fig.add_subplot(grid[1, 0])
    median = float(record_null["null_p50_gain"])
    low = float(record_null["null_p025_gain"])
    high = float(record_null["null_p975_gain"])
    observed = float(record_null["observed_gain"])
    null_ax.hlines(0, low, high, color=NO_CHANGE, linewidth=8, alpha=0.42)
    null_ax.scatter([median], [0], color=NO_CHANGE, s=75, marker="o", zorder=4)
    null_ax.scatter([observed], [0], color=EXACT, s=105, marker="D", zorder=5)
    null_ax.text(median, 0.16, f"null median {median:.2f}", ha="center", color=MUTED)
    null_ax.text(observed, 0.16, f"observed {observed:.2f}", ha="center", color=EXACT)
    null_ax.text(
        (low + high) / 2,
        -0.16,
        f"stationary schedule null 95%: {low:.2f}–{high:.2f}",
        ha="center",
        color=MUTED,
    )
    null_ax.set_ylim(-0.35, 0.38)
    null_ax.set_yticks([])
    null_ax.set_xlabel("Fixed-panel gain · score points")
    null_ax.set_title(
        f"C  More evaluations alone are insufficient · p={float(record_null['one_sided_p_value']):.5f}",
        loc="left",
    )
    clean_axis(null_ax, "x")

    # Panel D: observation schedules create material recovery error.
    sim_ax = fig.add_subplot(grid[1, 1])
    order = [
        "clean_logistic",
        "benchmark_heterogeneity",
        "heterogeneity_selective_stopping",
    ]
    sim_labels = [
        "Shared clock",
        "Heterogeneous clocks",
        "+ selective stopping",
    ]
    sim = simulation.set_index("scenario").loc[order]
    med = 100 * sim["median_curve_rmse_probability"].to_numpy()
    hi = 100 * sim["p975_curve_rmse_probability"].to_numpy()
    yy = np.arange(3)
    sim_ax.hlines(yy, med, hi, color=GRID, linewidth=4)
    sim_ax.scatter(med, yy, color=EXACT, marker="o", s=72, zorder=4)
    sim_ax.scatter(hi, yy, color=RIGHT, marker="|", s=150, linewidth=2, zorder=4)
    for value, row in zip(med, yy):
        sim_ax.text(
            value + 0.05,
            row,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=8.5,
        )
    sim_ax.set_yticks(yy)
    sim_ax.set_yticklabels(sim_labels)
    sim_ax.invert_yaxis()
    sim_ax.set_xlabel("Recovered-curve RMSE · percentage points")
    sim_ax.set_title("D  The empirical observation clock adds error", loc="left")
    sim_ax.text(
        0.98,
        0.04,
        "dot: median   tick: 97.5th percentile",
        transform=sim_ax.transAxes,
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    clean_axis(sim_ax, "x")

    fig.suptitle(
        "What survives—and what remains a measurement problem",
        x=0.06,
        ha="left",
        fontsize=18,
        fontweight="semibold",
        y=0.99,
    )
    fig.text(
        0.06,
        0.955,
        "The robust result is accumulated first-hit mass, not a unique sigmoid or a precisely estimated ceiling.",
        ha="left",
        color=MUTED,
        fontsize=10.5,
    )
    save(fig, "robustness_diagnostics.png")


def main() -> None:
    set_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure_measurement()
    figure_forecasts()
    figure_robustness()
    print("Wrote first_hit_measurement.png, forecast_evidence.png, robustness_diagnostics.png")


if __name__ == "__main__":
    main()
