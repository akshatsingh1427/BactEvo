"""
Bacterial Evolution Report Generator
Produces a fully self-contained HTML report with every meaningful chart
variation derived from the simulation_metrics.csv columns:
  time_step, total_population, resource_concentration,
  genotype_A_density, genotype_B_density, genotype_C_density,
  mutation_frequency, cooperation_index, competition_index
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
import base64
import os
from io import BytesIO
from datetime import datetime

# ── Theme ─────────────────────────────────────────────────────────────────────
DARK_BG    = "#0c1318"
SURF2      = "#111b22"
GRID_COLOR = "#1a2d38"
TEXT_COLOR = "#d4eaf7"
MUTED      = "#4a7a94"

C = {
    "green":  "#00e5a0",
    "blue":   "#00b8ff",
    "red":    "#ff4d6d",
    "yellow": "#ffd166",
    "purple": "#c77dff",
    "orange": "#ff9f43",
    "pink":   "#f72585",
    "teal":   "#4cc9f0",
}

GENOTYPE_COLORS = [C["red"], C["yellow"], C["green"]]
GENOTYPE_LABELS = ["Genotype A", "Genotype B", "Genotype C"]
GENOTYPE_COLS   = ["genotype_A_density", "genotype_B_density", "genotype_C_density"]


def style(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def b64(fig, name=None):
    """
    Converts figure to base64 for HTML AND optionally saves PNG.
    """

    if name:
        os.makedirs("charts", exist_ok=True)
        fig.savefig(
            os.path.join("charts", f"{name}.png"),
            dpi=140,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)

    enc = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return enc


def legend(ax, **kw):
    return ax.legend(frameon=False, fontsize=7.5, labelcolor=TEXT_COLOR, **kw)


def fmt_pop(x, _):
    if x >= 1e9: return f"{x/1e9:.1f}B"
    if x >= 1e6: return f"{x/1e6:.0f}M"
    if x >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — POPULATION
# ══════════════════════════════════════════════════════════════════════════════

def chart_population_line(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    clr = C["green"]
    ax.fill_between(d.time_step, d.total_population, alpha=0.13, color=clr)
    ax.plot(d.time_step, d.total_population, color=clr, lw=1.8)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Total Population")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_pop))
    fig.tight_layout(pad=1.4)
    return b64(fig, "population_line")


def chart_population_rolling(d, window=20):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    clr = C["green"]
    roll = d.total_population.rolling(window, min_periods=1).mean()
    ax.fill_between(d.time_step, d.total_population, alpha=0.08, color=clr)
    ax.plot(d.time_step, d.total_population, color=clr, lw=0.8, alpha=0.4, label="Raw")
    ax.plot(d.time_step, roll, color=clr, lw=2, label=f"Rolling mean ({window})")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_pop))
    ax.set_xlabel("Time Step"); ax.set_ylabel("Total Population")
    legend(ax)
    fig.tight_layout(pad=1.4)
    return b64(fig, "population_rolling")


def chart_population_growth_rate(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    gr = d.total_population.pct_change() * 100
    clr = C["teal"]
    ax.axhline(0, color=GRID_COLOR, lw=1)
    ax.fill_between(d.time_step, gr, where=(gr >= 0), alpha=0.2, color=C["green"])
    ax.fill_between(d.time_step, gr, where=(gr < 0),  alpha=0.2, color=C["red"])
    ax.plot(d.time_step, gr, color=clr, lw=1.2)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Growth Rate (%)")
    fig.tight_layout(pad=1.4)
    return b64(fig, "population_growth_rate")


def chart_population_log(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    clr = C["green"]
    vals = d.total_population.clip(lower=1)
    ax.semilogy(d.time_step, vals, color=clr, lw=1.8)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Population (log scale)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_pop))
    fig.tight_layout(pad=1.4)
    return b64(fig, "population_log")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — RESOURCE
# ══════════════════════════════════════════════════════════════════════════════

def chart_resource_line(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    clr = C["blue"]
    ax.fill_between(d.time_step, d.resource_concentration, alpha=0.13, color=clr)
    ax.plot(d.time_step, d.resource_concentration, color=clr, lw=1.8)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Resource Conc. (mM)")
    fig.tight_layout(pad=1.4)
    return b64(fig, "resource_line")


def chart_resource_vs_population(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    sc = ax.scatter(d.resource_concentration, d.total_population,
                    c=d.time_step, cmap="cool", s=8, alpha=0.7, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Time Step", color=MUTED, fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_pop))
    ax.set_xlabel("Resource Conc. (mM)"); ax.set_ylabel("Total Population")
    fig.tight_layout(pad=1.4)
    return b64(fig, "resource_vs_population")


def chart_resource_depletion_rate(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    dr = d.resource_concentration.diff().fillna(0)
    clr = C["blue"]
    ax.axhline(0, color=GRID_COLOR, lw=1)
    ax.fill_between(d.time_step, dr, where=(dr >= 0), alpha=0.2, color=C["teal"])
    ax.fill_between(d.time_step, dr, where=(dr < 0),  alpha=0.2, color=C["red"])
    ax.plot(d.time_step, dr, color=clr, lw=1)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Δ Resource / Step")
    fig.tight_layout(pad=1.4)
    return b64(fig, "resource_depletion_rate")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — GENOTYPES
# ══════════════════════════════════════════════════════════════════════════════

def chart_genotypes_lines(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    for col, lbl, clr in zip(GENOTYPE_COLS, GENOTYPE_LABELS, GENOTYPE_COLORS):
        ax.fill_between(d.time_step, d[col], alpha=0.08, color=clr)
        ax.plot(d.time_step, d[col], label=lbl, color=clr, lw=1.6)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Density")
    legend(ax, loc="upper right")
    fig.tight_layout(pad=1.4)
    return b64(fig, "genotypes_lines")


def chart_genotypes_stacked_area(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    ax.stackplot(d.time_step,
                 d.genotype_A_density, d.genotype_B_density, d.genotype_C_density,
                 labels=GENOTYPE_LABELS, colors=GENOTYPE_COLORS, alpha=0.75)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Cumulative Density")
    legend(ax, loc="upper left")
    fig.tight_layout(pad=1.4)
    return b64(fig, "genotypes_stacked_area")


def chart_genotypes_proportion(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    total = d[GENOTYPE_COLS].sum(axis=1).replace(0, np.nan)
    props = [d[c] / total * 100 for c in GENOTYPE_COLS]
    ax.stackplot(d.time_step, *props,
                 labels=GENOTYPE_LABELS, colors=GENOTYPE_COLORS, alpha=0.75)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_xlabel("Time Step"); ax.set_ylabel("Proportion (%)")
    legend(ax, loc="upper left")
    fig.tight_layout(pad=1.4)
    return b64(fig, "genotypes_proportion")


def chart_genotypes_scatter_matrix(d):
    sub = d[GENOTYPE_COLS].rename(columns={
        "genotype_A_density": "Gen A",
        "genotype_B_density": "Gen B",
        "genotype_C_density": "Gen C",
    })
    g = sns.PairGrid(sub, height=1.8, aspect=1)
    g.fig.patch.set_facecolor(DARK_BG)
    g.map_upper(sns.scatterplot, s=6, alpha=0.5, color=C["teal"])
    g.map_lower(sns.kdeplot, fill=True, color=C["purple"], alpha=0.4)
    g.map_diag(sns.histplot, color=C["yellow"], alpha=0.6, edgecolor="none")
    for ax in g.axes.flat:
        style(ax, g.fig)
    g.fig.tight_layout(pad=1)
    return b64(g.fig, "genotypes_scatter_matrix")


def chart_genotypes_boxplot(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    vals = [d[c].dropna() for c in GENOTYPE_COLS]
    bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                    medianprops=dict(color="#fff", lw=2),
                    whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", markersize=2, color=MUTED, alpha=0.5))
    for patch, clr in zip(bp["boxes"], GENOTYPE_COLORS):
        patch.set_facecolor(clr); patch.set_alpha(0.4)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(GENOTYPE_LABELS, color=MUTED, fontsize=8)
    ax.set_ylabel("Density")
    fig.tight_layout(pad=1.4)
    return b64(fig, "genotypes_boxplot")


def chart_genotype_dominance(d):
    fig, ax = plt.subplots(figsize=(6, 1.6))
    style(ax, fig)
    dom = d[GENOTYPE_COLS].idxmax(axis=1).map({
        "genotype_A_density": 0,
        "genotype_B_density": 1,
        "genotype_C_density": 2,
    })
    cmap = matplotlib.colors.ListedColormap(GENOTYPE_COLORS)
    ax.scatter(d.time_step, np.zeros(len(d)), c=dom, cmap=cmap, s=6, marker="|")
    patches = [mpatches.Patch(color=clr, label=lbl)
               for clr, lbl in zip(GENOTYPE_COLORS, GENOTYPE_LABELS)]
    ax.legend(handles=patches, frameon=False, fontsize=7.5,
              labelcolor=TEXT_COLOR, loc="upper right", ncol=3)
    ax.set_xlabel("Time Step")
    ax.set_yticks([])
    ax.set_ylabel("Dominant", color=MUTED, fontsize=8)
    fig.tight_layout(pad=1.4)
    return b64(fig, "genotype_dominance")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MUTATION
# ══════════════════════════════════════════════════════════════════════════════

def chart_mutation_hist(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    clr = C["yellow"]
    sns.histplot(d.mutation_frequency, kde=True, ax=ax,
                 color=clr, alpha=0.25, edgecolor="none",
                 line_kws={"lw": 2, "color": clr})
    for line in ax.lines: line.set_color(clr); line.set_lw(2)
    ax.set_xlabel("Mutation Frequency"); ax.set_ylabel("Count")
    fig.tight_layout(pad=1.4)
    return b64(fig, "mutation_histogram")


def chart_mutation_over_time(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    clr = C["yellow"]
    ax.fill_between(d.time_step, d.mutation_frequency, alpha=0.13, color=clr)
    ax.plot(d.time_step, d.mutation_frequency, color=clr, lw=1.6)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Mutation Frequency")
    fig.tight_layout(pad=1.4)
    return b64(fig, "mutation_over_time")


def chart_mutation_violin(d):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    style(ax, fig)
    parts = ax.violinplot(d.mutation_frequency.dropna(), positions=[0],
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(C["yellow"]); pc.set_alpha(0.45)
    parts["cmedians"].set_color(TEXT_COLOR)
    parts["cmins"].set_color(MUTED); parts["cmaxes"].set_color(MUTED)
    parts["cbars"].set_color(MUTED)
    ax.set_xticks([]); ax.set_ylabel("Mutation Frequency")
    fig.tight_layout(pad=1.4)
    return b64(fig, "mutation_violin")


def chart_mutation_vs_population(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    sc = ax.scatter(d.mutation_frequency, d.total_population,
                    c=d.time_step, cmap="YlOrRd", s=8, alpha=0.6, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Time Step", color=MUTED, fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_pop))
    ax.set_xlabel("Mutation Frequency"); ax.set_ylabel("Total Population")
    fig.tight_layout(pad=1.4)
    return b64(fig, "mutation_vs_population")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — COOPERATION & COMPETITION
# ══════════════════════════════════════════════════════════════════════════════

def chart_coop_comp_lines(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    ax.fill_between(d.time_step, d.cooperation_index, alpha=0.10, color=C["green"])
    ax.fill_between(d.time_step, d.competition_index, alpha=0.10, color=C["red"])
    ax.plot(d.time_step, d.cooperation_index, color=C["green"], lw=1.8, label="Cooperation")
    ax.plot(d.time_step, d.competition_index, color=C["red"],   lw=1.8, label="Competition")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Index")
    legend(ax, loc="best")
    fig.tight_layout(pad=1.4)
    return b64(fig, "coop_comp_lines")


def chart_coop_comp_ratio(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    ratio = d.cooperation_index / d.competition_index.replace(0, np.nan)
    clr = C["purple"]
    ax.axhline(1.0, color=MUTED, lw=1, linestyle=":")
    ax.fill_between(d.time_step, ratio, 1, where=(ratio >= 1),
                    alpha=0.2, color=C["green"], interpolate=True)
    ax.fill_between(d.time_step, ratio, 1, where=(ratio < 1),
                    alpha=0.2, color=C["red"], interpolate=True)
    ax.plot(d.time_step, ratio, color=clr, lw=1.6)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Cooperation / Competition")
    fig.tight_layout(pad=1.4)
    return b64(fig, "coop_comp_ratio")


def chart_coop_comp_scatter(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    sc = ax.scatter(d.competition_index, d.cooperation_index,
                    c=d.time_step, cmap="plasma", s=8, alpha=0.65, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Time Step", color=MUTED, fontsize=8)
    ax.set_xlabel("Competition Index"); ax.set_ylabel("Cooperation Index")
    lo = min(d.competition_index.min(), d.cooperation_index.min())
    hi = max(d.competition_index.max(), d.cooperation_index.max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, linestyle=":")
    fig.tight_layout(pad=1.4)
    return b64(fig, "coop_comp_scatter")


def chart_coop_comp_hist(d):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    for ax, col, clr, lbl in [
        (axes[0], "cooperation_index", C["green"], "Cooperation"),
        (axes[1], "competition_index", C["red"],   "Competition"),
    ]:
        style(ax, fig)
        sns.histplot(d[col], kde=True, ax=ax, color=clr, alpha=0.25,
                     edgecolor="none", line_kws={"lw": 2, "color": clr})
        for line in ax.lines: line.set_color(clr); line.set_lw(2)
        ax.set_xlabel(lbl); ax.set_ylabel("Count" if ax is axes[0] else "")
    fig.tight_layout(pad=1.4)
    return b64(fig, "coop_comp_hist")


def chart_coop_vs_mutation(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    sc = ax.scatter(d.mutation_frequency, d.cooperation_index,
                    c=d.time_step, cmap="cool", s=8, alpha=0.65, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Time Step", color=MUTED, fontsize=8)
    ax.set_xlabel("Mutation Frequency"); ax.set_ylabel("Cooperation Index")
    fig.tight_layout(pad=1.4)
    return b64(fig, "coop_vs_mutation")


def chart_comp_vs_mutation(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    sc = ax.scatter(d.mutation_frequency, d.competition_index,
                    c=d.time_step, cmap="autumn", s=8, alpha=0.65, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Time Step", color=MUTED, fontsize=8)
    ax.set_xlabel("Mutation Frequency"); ax.set_ylabel("Competition Index")
    fig.tight_layout(pad=1.4)
    return b64(fig, "comp_vs_mutation")


def chart_rolling_coop_comp(d, window=20):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    for col, lbl, clr in [
        ("cooperation_index", "Cooperation", C["green"]),
        ("competition_index", "Competition", C["red"]),
    ]:
        raw  = d[col]
        roll = raw.rolling(window, min_periods=1).mean()
        ax.plot(d.time_step, raw,  color=clr, lw=0.6, alpha=0.3)
        ax.plot(d.time_step, roll, color=clr, lw=2.0, label=f"{lbl} (rolling {window})")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Index")
    legend(ax)
    fig.tight_layout(pad=1.4)
    return b64(fig, "rolling_coop_comp")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CROSS-VARIABLE
# ══════════════════════════════════════════════════════════════════════════════

def chart_correlation_heatmap(d):
    cols = ["total_population", "resource_concentration",
            "genotype_A_density", "genotype_B_density", "genotype_C_density",
            "mutation_frequency", "cooperation_index", "competition_index"]
    short = ["Pop", "Resource", "GenA", "GenB", "GenC", "Mutation", "Coop", "Comp"]
    corr = d[cols].corr()
    corr.index = short; corr.columns = short
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    style(ax, fig)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", annot_kws={"size": 7},
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor=DARK_BG,
                cbar_kws={"shrink": 0.75})
    ax.tick_params(colors=TEXT_COLOR, labelsize=8, rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout(pad=1.2)
    return b64(fig, "correlation_heatmap")


def chart_all_series_normalised(d):
    cols = ["total_population", "resource_concentration",
            "genotype_A_density", "genotype_B_density", "genotype_C_density",
            "mutation_frequency", "cooperation_index", "competition_index"]
    labels = ["Population", "Resource", "GenA", "GenB", "GenC",
              "Mutation", "Cooperation", "Competition"]
    colors = [C["green"], C["blue"], C["red"], C["yellow"], C["teal"],
              C["orange"], C["purple"], C["pink"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    style(ax, fig)
    for col, lbl, clr in zip(cols, labels, colors):
        s = d[col]
        rng = s.max() - s.min()
        norm = (s - s.min()) / rng if rng != 0 else s * 0
        ax.plot(d.time_step, norm, label=lbl, color=clr, lw=1.2, alpha=0.85)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Normalised Value (0–1)")
    legend(ax, loc="upper right", ncol=2)
    fig.tight_layout(pad=1.4)
    return b64(fig, "all_series_normalised")


def chart_population_resource_dual_axis(d):
    fig, ax1 = plt.subplots(figsize=(6, 3.2))
    style(ax1, fig)
    ax2 = ax1.twinx()
    style(ax2, fig)
    ax2.set_facecolor("none")
    ax1.fill_between(d.time_step, d.total_population, alpha=0.10, color=C["green"])
    l1, = ax1.plot(d.time_step, d.total_population, color=C["green"], lw=1.8, label="Population")
    l2, = ax2.plot(d.time_step, d.resource_concentration, color=C["blue"], lw=1.8, linestyle="--", label="Resource")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_pop))
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Population", color=C["green"])
    ax2.set_ylabel("Resource Conc. (mM)", color=C["blue"])
    ax1.tick_params(axis="y", colors=C["green"])
    ax2.tick_params(axis="y", colors=C["blue"])
    ax1.legend(handles=[l1, l2], frameon=False, fontsize=7.5,
               labelcolor=TEXT_COLOR, loc="upper right")
    fig.tight_layout(pad=1.4)
    return b64(fig, "population_resource_dual_axis")


def chart_coop_comp_population(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    for col, lbl, clr in [
        ("total_population",  "Population",   C["green"]),
        ("cooperation_index", "Cooperation",  C["blue"]),
        ("competition_index", "Competition",  C["red"]),
    ]:
        s = d[col]
        rng = s.max() - s.min()
        norm = (s - s.min()) / rng if rng != 0 else s * 0
        ax.plot(d.time_step, norm, label=lbl, color=clr, lw=1.5)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Normalised Value")
    legend(ax, loc="best")
    fig.tight_layout(pad=1.4)
    return b64(fig, "coop_comp_population")


def chart_genotype_resource_scatter(d):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    total_gen = d[GENOTYPE_COLS].sum(axis=1)
    sc = ax.scatter(d.resource_concentration, total_gen,
                    c=d.time_step, cmap="viridis", s=8, alpha=0.6, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Time Step", color=MUTED, fontsize=8)
    ax.set_xlabel("Resource Conc. (mM)"); ax.set_ylabel("Total Genotype Density")
    fig.tight_layout(pad=1.4)
    return b64(fig, "genotype_resource_scatter")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SIMULATION PARAMETERS & ENVIRONMENT  ← NEW
# ══════════════════════════════════════════════════════════════════════════════

def chart_environment_resource_regime(d):
    """Highlights resource-rich vs resource-depleted periods relative to median."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    threshold = d.resource_concentration.median()
    rich     = d.resource_concentration.where(d.resource_concentration >= threshold)
    depleted = d.resource_concentration.where(d.resource_concentration < threshold)
    ax.fill_between(d.time_step, rich,     alpha=0.30, color=C["blue"],  label="Resource-rich")
    ax.fill_between(d.time_step, depleted, alpha=0.30, color=C["red"],   label="Resource-depleted")
    ax.axhline(threshold, color=MUTED, lw=1, linestyle=":", label=f"Median ({threshold:.3f} mM)")
    ax.plot(d.time_step, d.resource_concentration, color=C["teal"], lw=1.2, alpha=0.8)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Resource Conc. (mM)")
    legend(ax, loc="upper right")
    fig.tight_layout(pad=1.4)
    return b64(fig, "environment_resource_regime")


def chart_antibiotic_stress_proxy(d):
    """Uses mutation_frequency spikes as proxy for antibiotic / stress events (gradual or sudden)."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    mu           = d.mutation_frequency
    spike_thresh = mu.mean() + 1.5 * mu.std()
    spikes       = mu >= spike_thresh
    ax.fill_between(d.time_step, mu, alpha=0.12, color=C["yellow"])
    ax.plot(d.time_step, mu, color=C["yellow"], lw=1.4, label="Mutation Freq")
    ax.axhline(spike_thresh, color=C["red"], lw=1, linestyle="--", label="Spike threshold (μ + 1.5σ)")
    ax.fill_between(d.time_step, mu, spike_thresh,
                    where=spikes, alpha=0.35, color=C["red"],
                    interpolate=True, label="Antibiotic / stress events")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Mutation Frequency")
    legend(ax, loc="upper right")
    fig.tight_layout(pad=1.4)
    return b64(fig, "antibiotic_stress_proxy")


def chart_fitness_landscape_proxy(d):
    """Fitness proxy = pop growth rate normalised by resource — reveals landscape curvature."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    gr      = d.total_population.pct_change().fillna(0)
    res     = d.resource_concentration.replace(0, np.nan)
    fitness = (gr / res).clip(-5, 5)
    ax.axhline(0, color=MUTED, lw=1)
    ax.fill_between(d.time_step, fitness, where=(fitness >= 0), alpha=0.22, color=C["green"])
    ax.fill_between(d.time_step, fitness, where=(fitness < 0),  alpha=0.22, color=C["red"])
    ax.plot(d.time_step, fitness, color=C["purple"], lw=1.4, label="Fitness proxy (ΔPop / Resource)")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Fitness Proxy")
    legend(ax)
    fig.tight_layout(pad=1.4)
    return b64(fig, "fitness_landscape_proxy")


def chart_carrying_capacity_saturation(d):
    """Population as % of running maximum — shows proximity to carrying capacity K."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    running_max = d.total_population.cummax()
    saturation  = d.total_population / running_max.replace(0, np.nan) * 100
    ax.fill_between(d.time_step, saturation, alpha=0.15, color=C["orange"])
    ax.plot(d.time_step, saturation, color=C["orange"], lw=1.8, label="K-saturation (%)")
    ax.axhline(90, color=C["red"],    lw=1, linestyle=":", label="90% K")
    ax.axhline(50, color=C["yellow"], lw=1, linestyle=":", label="50% K")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_xlabel("Time Step"); ax.set_ylabel("% of Running Max (K-proxy)")
    legend(ax, loc="lower right")
    fig.tight_layout(pad=1.4)
    return b64(fig, "carrying_capacity_saturation")


def chart_generation_time_proxy(d):
    """Estimates inverse generation time from log₂ population doublings per step."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    pop           = d.total_population.clip(lower=1)
    log_pop       = np.log2(pop)
    doubling_rate = log_pop.diff().fillna(0).clip(lower=0)
    roll          = doubling_rate.rolling(15, min_periods=1).mean()
    ax.fill_between(d.time_step, doubling_rate, alpha=0.12, color=C["teal"])
    ax.plot(d.time_step, doubling_rate, color=C["teal"], lw=0.8, alpha=0.4, label="Raw doublings / step")
    ax.plot(d.time_step, roll,          color=C["teal"], lw=2.0,            label="Rolling mean (15 steps)")
    ax.set_xlabel("Time Step"); ax.set_ylabel("log₂ Doublings / Step")
    legend(ax)
    fig.tight_layout(pad=1.4)
    return b64(fig, "generation_time_proxy")


def chart_adversarial_dynamics(d):
    """Adversarial pressure: competition index + resource depletion rate on dual axes."""
    fig, ax1 = plt.subplots(figsize=(6, 3.2))
    style(ax1, fig)
    ax2 = ax1.twinx()
    style(ax2, fig); ax2.set_facecolor("none")
    dr = d.resource_concentration.diff().fillna(0)
    l1, = ax1.plot(d.time_step, d.competition_index,
                   color=C["red"],    lw=1.8,              label="Competition Index")
    l2, = ax2.plot(d.time_step, dr,
                   color=C["orange"], lw=1.2, linestyle="--", alpha=0.75, label="Δ Resource / Step")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Competition Index", color=C["red"])
    ax2.set_ylabel("Δ Resource / Step",  color=C["orange"])
    ax1.tick_params(axis="y", colors=C["red"])
    ax2.tick_params(axis="y", colors=C["orange"])
    ax1.legend(handles=[l1, l2], frameon=False, fontsize=7.5,
               labelcolor=TEXT_COLOR, loc="upper right")
    fig.tight_layout(pad=1.4)
    return b64(fig, "adversarial_dynamics")


def chart_cooperative_behaviors(d):
    """Public-good / biofilm proxy: cooperation index vs Shannon genotype diversity."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    total_gen = d[GENOTYPE_COLS].sum(axis=1).replace(0, np.nan)
    props     = [d[c] / total_gen for c in GENOTYPE_COLS]
    diversity = -(sum(p * np.log(p.clip(lower=1e-9)) for p in props))   # Shannon entropy
    div_norm  = (diversity - diversity.min()) / (diversity.max() - diversity.min() + 1e-12)
    coop_norm = (d.cooperation_index - d.cooperation_index.min()) / \
                (d.cooperation_index.max()  - d.cooperation_index.min()  + 1e-12)
    ax.fill_between(d.time_step, coop_norm, alpha=0.12, color=C["green"])
    ax.plot(d.time_step, coop_norm, color=C["green"],  lw=1.6,              label="Cooperation (norm.)")
    ax.plot(d.time_step, div_norm,  color=C["purple"], lw=1.4, linestyle="--",
            label="Genotype Diversity — Shannon (norm.)")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Normalised Value")
    legend(ax, loc="best")
    fig.tight_layout(pad=1.4)
    return b64(fig, "cooperative_behaviors")


def chart_hgt_quorum_proxy(d):
    """HGT / quorum sensing proxy: cooperation surges when population density is high."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    pop_norm  = (d.total_population   - d.total_population.min())   / \
                (d.total_population.max()   - d.total_population.min()   + 1e-12)
    coop_norm = (d.cooperation_index  - d.cooperation_index.min())  / \
                (d.cooperation_index.max()  - d.cooperation_index.min()  + 1e-12)
    sc = ax.scatter(pop_norm, coop_norm, c=d.time_step, cmap="cool",
                    s=8, alpha=0.65, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Time Step", color=MUTED, fontsize=8)
    ax.set_xlabel("Population Density (norm.)  ← Quorum proxy")
    ax.set_ylabel("Cooperation Index (norm.)   ← HGT proxy")
    fig.tight_layout(pad=1.4)
    return b64(fig, "hgt_quorum_proxy")


def chart_spatial_genotype_heatmap(d):
    """Pseudo-spatial heatmap: time × genotype — simulated 1-D spatial distribution proxy."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    style(ax, fig)
    mat = d[GENOTYPE_COLS].T.values
    im  = ax.imshow(mat, aspect="auto", cmap="inferno",
                    extent=[d.time_step.iloc[0], d.time_step.iloc[-1], 0, 3],
                    origin="lower", interpolation="bilinear")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Density", color=MUTED, fontsize=8)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Gen A", "Gen B", "Gen C"], color=MUTED, fontsize=8)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Genotype")
    fig.tight_layout(pad=1.4)
    return b64(fig, "spatial_genotype_heatmap")


def chart_evolutionary_driver_radar(d):
    """Radar / spider chart: 6-axis normalised mean of key evolutionary drivers."""
    categories = ["Mutation\nRate", "Cooperation", "Competition",
                  "Resource\nDepletion", "Pop\nGrowth", "Diversity"]

    mu       = d.mutation_frequency.mean()
    coop     = d.cooperation_index.mean()
    comp     = d.competition_index.mean()
    deplete  = (-d.resource_concentration.diff().fillna(0)).clip(lower=0).mean()
    growth   = d.total_population.pct_change().fillna(0).clip(lower=0).mean()
    total_g  = d[GENOTYPE_COLS].sum(axis=1).replace(0, np.nan)
    props    = [d[c] / total_g for c in GENOTYPE_COLS]
    diversity = -(sum(p * np.log(p.clip(lower=1e-9)) for p in props)).mean()

    raw  = np.array([mu, coop, comp, deplete, growth, diversity])
    raw  = raw / (raw.max() + 1e-12)
    vals = np.concatenate([raw, [raw[0]]])

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.spines["polar"].set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT_COLOR, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=MUTED, fontsize=6)

    ax.plot(angles, vals, color=C["green"], lw=2)
    ax.fill(angles, vals, color=C["green"], alpha=0.18)
    for ang, val in zip(angles[:-1], raw):
        ax.plot(ang, val, "o", color=C["green"], ms=5)

    fig.tight_layout(pad=1.2)
    return b64(fig, "evolutionary_driver_radar")


# ══════════════════════════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(d):
    mean_comp = d.competition_index.mean()
    ratio = f"{d.cooperation_index.mean()/mean_comp:.2f}" if mean_comp != 0 else "N/A"
    threshold    = d.resource_concentration.median()
    rich_pct     = (d.resource_concentration >= threshold).mean() * 100
    mu           = d.mutation_frequency
    spike_thresh = mu.mean() + 1.5 * mu.std()
    stress_events = int((mu >= spike_thresh).sum())
    return {
        "max_pop":         f"{d.total_population.max():,.0f}",
        "min_resource":    f"{d.resource_concentration.min():.4f} mM",
        "mean_mutation":   f"{d.mutation_frequency.mean():.2e}",
        "mean_coop":       f"{d.cooperation_index.mean():.3f}",
        "mean_comp":       f"{d.competition_index.mean():.3f}",
        "num_genotypes":   "3",
        "time_steps":      f"{len(d):,}",
        "peak_step":       f"{d.loc[d.total_population.idxmax(), 'time_step']:,}",
        "gen_date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "coop_comp_ratio": ratio,
        "rich_pct":        f"{rich_pct:.1f}%",
        "stress_events":   f"{stress_events}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HTML
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
:root{
  --bg:#060a0f;--surface:#0c1318;--surf2:#111b22;--border:#1a2d38;
  --green:#00e5a0;--blue:#00b8ff;--red:#ff4d6d;--yellow:#ffd166;
  --purple:#c77dff;--orange:#ff9f43;--pink:#f72585;--teal:#4cc9f0;
  --text:#d4eaf7;--muted:#4a7a94;
  --mono:'Space Mono',monospace;--display:'Syne',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--mono);overflow-x:hidden}
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(0,229,160,.025) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,229,160,.025) 1px,transparent 1px);
  background-size:44px 44px;
}
nav{
  position:sticky;top:0;z-index:100;
  background:rgba(6,10,15,.88);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
  padding:0 60px;display:flex;align-items:center;gap:4px;overflow-x:auto;
}
nav a{
  font-size:9px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);padding:14px 12px;text-decoration:none;
  border-bottom:2px solid transparent;white-space:nowrap;
  transition:color .2s,border-color .2s;
}
nav a:hover{color:var(--text)}
nav a.active{color:var(--green);border-bottom-color:var(--green)}
header{
  position:relative;z-index:10;
  padding:60px 60px 40px;border-bottom:1px solid var(--border);
  display:grid;grid-template-columns:1fr auto;align-items:end;gap:40px;
  animation:fadeDown .7s ease both;
}
.tags{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:22px;align-items:center}
.tag{font-size:10px;letter-spacing:.14em;text-transform:uppercase;padding:4px 10px;border-radius:2px}
.tag.g{color:var(--green); background:rgba(0,229,160,.08); border:1px solid rgba(0,229,160,.2)}
.tag.b{color:var(--blue);  background:rgba(0,184,255,.08); border:1px solid rgba(0,184,255,.2)}
.tag.r{color:var(--red);   background:rgba(255,77,109,.08);border:1px solid rgba(255,77,109,.2)}
.tag.y{color:var(--yellow);background:rgba(255,209,102,.08);border:1px solid rgba(255,209,102,.2)}
.run-id{font-size:10px;color:var(--muted);letter-spacing:.1em}
h1{font-family:var(--display);font-size:clamp(34px,5vw,58px);font-weight:800;
   line-height:1.05;letter-spacing:-.02em;color:#fff}
h1 span{color:var(--green)}
.kpis{display:flex;flex-direction:column;gap:16px;align-items:flex-end}
.kpi{display:flex;flex-direction:column;align-items:flex-end;gap:3px}
.kpi .val{font-family:var(--display);font-size:24px;font-weight:700;color:#fff}
.kpi .lbl{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em}
.div-h{width:1px;height:32px;background:var(--border)}
section{position:relative;z-index:10;padding:50px 60px 10px}
.slabel{
  font-size:10px;color:var(--muted);letter-spacing:.18em;text-transform:uppercase;
  margin-bottom:8px;display:flex;align-items:center;gap:12px;
}
.slabel::after{content:'';flex:1;height:1px;background:var(--border)}
h2{font-family:var(--display);font-size:22px;font-weight:700;color:#fff;margin-bottom:28px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-bottom:22px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:22px;margin-bottom:22px}
.full{grid-column:1/-1}
.card{
  background:var(--surface);border:1px solid var(--border);border-radius:4px;
  overflow:hidden;position:relative;
  animation:fadeUp .65s ease both;
  transition:border-color .3s,box-shadow .3s;
}
.card:hover{border-color:rgba(0,229,160,.2);box-shadow:0 0 28px rgba(0,229,160,.04),0 14px 44px rgba(0,0,0,.4)}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.card.g::before{background:linear-gradient(90deg,var(--green),transparent)}
.card.b::before{background:linear-gradient(90deg,var(--blue),transparent)}
.card.r::before{background:linear-gradient(90deg,var(--red),transparent)}
.card.y::before{background:linear-gradient(90deg,var(--yellow),transparent)}
.card.p::before{background:linear-gradient(90deg,var(--purple),transparent)}
.card.o::before{background:linear-gradient(90deg,var(--orange),transparent)}
.card.t::before{background:linear-gradient(90deg,var(--teal),transparent)}
.d1{animation-delay:.05s}.d2{animation-delay:.12s}.d3{animation-delay:.19s}
.d4{animation-delay:.26s}.d5{animation-delay:.33s}.d6{animation-delay:.40s}
.d7{animation-delay:.47s}.d8{animation-delay:.54s}
.card-head{
  padding:16px 20px 12px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:flex-start;gap:10px;
}
.card-meta{display:flex;flex-direction:column;gap:4px}
.card-num{font-size:9px;letter-spacing:.15em}
.card.g .card-num{color:var(--green)}.card.b .card-num{color:var(--blue)}
.card.r .card-num{color:var(--red)}.card.y .card-num{color:var(--yellow)}
.card.p .card-num{color:var(--purple)}.card.o .card-num{color:var(--orange)}
.card.t .card-num{color:var(--teal)}
.card-title{font-family:var(--display);font-size:14px;font-weight:700;color:#fff}
.badge{font-size:8px;text-transform:uppercase;letter-spacing:.12em;
       padding:2px 7px;border-radius:2px;white-space:nowrap;margin-top:2px;display:inline-block}
.card.g .badge{color:var(--green); background:rgba(0,229,160,.08); border:1px solid rgba(0,229,160,.15)}
.card.b .badge{color:var(--blue);  background:rgba(0,184,255,.08); border:1px solid rgba(0,184,255,.15)}
.card.r .badge{color:var(--red);   background:rgba(255,77,109,.08);border:1px solid rgba(255,77,109,.15)}
.card.y .badge{color:var(--yellow);background:rgba(255,209,102,.08);border:1px solid rgba(255,209,102,.15)}
.card.p .badge{color:var(--purple);background:rgba(199,125,255,.08);border:1px solid rgba(199,125,255,.15)}
.card.o .badge{color:var(--orange);background:rgba(255,159,67,.08); border:1px solid rgba(255,159,67,.15)}
.card.t .badge{color:var(--teal);  background:rgba(76,201,240,.08); border:1px solid rgba(76,201,240,.15)}
.card-note{font-size:9px;color:var(--muted);line-height:1.6;max-width:135px;text-align:right}
.chart-body{padding:16px 18px;background:var(--surf2)}
.chart-body img{width:100%;height:auto;display:block;border-radius:2px}
.card-foot{padding:9px 20px;border-top:1px solid var(--border);
           display:flex;justify-content:space-between;align-items:center}
.metric{display:flex;flex-direction:column;gap:2px}
.metric .mv{font-size:12px;font-weight:700;color:#fff}
.metric .ml{font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.trend{font-size:9px;padding:2px 8px;border-radius:20px;display:flex;align-items:center;gap:3px}
.trend.up  {color:var(--green); background:rgba(0,229,160,.1)}
.trend.dn  {color:var(--red);   background:rgba(255,77,109,.1)}
.trend.flat{color:var(--yellow);background:rgba(255,209,102,.1)}
.trend.mix {color:var(--purple);background:rgba(199,125,255,.1)}
footer{
  position:relative;z-index:10;border-top:1px solid var(--border);
  padding:22px 60px;display:flex;justify-content:space-between;align-items:center;
  animation:fadeUp .7s .6s ease both;
}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
.foot-l{display:flex;align-items:center;gap:14px}
.foot-txt,.foot-r{font-size:9px;color:var(--muted);letter-spacing:.1em}
@keyframes fadeUp  {from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeDown{from{opacity:0;transform:translateY(-12px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,229,160,.4)}50%{opacity:.7;box-shadow:0 0 0 6px rgba(0,229,160,0)}}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
@media(max-width:900px){
  nav{padding:0 14px}
  header{padding:30px 18px 22px;grid-template-columns:1fr}
  section{padding:34px 18px 8px}
  .grid-2,.grid-3{grid-template-columns:1fr}
  footer{padding:16px 18px;flex-direction:column;gap:10px}
  .kpis{flex-direction:row}
}
"""

def mk_card(idx, total, color, title, badge, note, img, mv, ml, tcls, ttxt, delay):
    return f"""
<div class="card {color} d{delay}">
  <div class="card-head">
    <div class="card-meta">
      <span class="card-num">{idx:02d}&thinsp;/&thinsp;{total:02d}</span>
      <span class="card-title">{title}</span>
      <span class="badge">{badge}</span>
    </div>
    <div class="card-note">{note}</div>
  </div>
  <div class="chart-body"><img src="data:image/png;base64,{img}" alt="{title}"></div>
  <div class="card-foot">
    <div class="metric"><span class="mv">{mv}</span><span class="ml">{ml}</span></div>
    <span class="trend {tcls}">{ttxt}</span>
  </div>
</div>"""


def build_html(imgs, s):
    def sec(sid, lbl, title, grid, cards):
        return f"""
<section id="{sid}">
  <div class="slabel">{lbl}</div>
  <h2>{title}</h2>
  <div class="{grid}">{"".join(cards)}</div>
</section>"""

    pop = [
        mk_card(1,4,"g","Population Over Time","Line Chart","Filled-area raw trajectory",imgs["pop_line"],s["max_pop"],"Max Population","up","↑ Growing",1),
        mk_card(2,4,"g","Rolling Mean Overlay","Smoothed Trend","Raw signal + rolling average",imgs["pop_roll"],s["peak_step"],"Peak Step","up","↑ Trend",2),
        mk_card(3,4,"t","Growth Rate","% Change / Step","Green=growth · Red=decline",imgs["pop_gr"],"—","Δ per step","mix","± Variable",3),
        mk_card(4,4,"g","Log Scale","Exponential View","Reveals multiplicative dynamics",imgs["pop_log"],s["max_pop"],"Max Pop","up","↑ Log",4),
    ]
    res = [
        mk_card(1,3,"b","Resource Concentration","Time Series","Nutrient level over time",imgs["res_line"],s["min_resource"],"Min Conc.","dn","↓ Depleting",1),
        mk_card(2,3,"b","Depletion Rate","Δ per Step","Rate of nutrient consumption",imgs["res_dr"],"—","Rate","dn","↓ Rate",2),
        mk_card(3,3,"b","Resource vs Population","Scatter","Colour = time progression",imgs["res_vs_pop"],s["max_pop"],"At Min Res","mix","~ Coupled",3),
    ]
    gen = [
        mk_card(1,6,"r","Genotype Densities","Multi-line","All three lineages over time",imgs["gen_lines"],"3","Genotypes","dn","↓ Diversity",1),
        mk_card(2,6,"r","Stacked Area","Composition","Absolute cumulative density",imgs["gen_stack"],"A+B+C","Combined","mix","± Competing",2),
        mk_card(3,6,"y","Relative Proportion","100% Stacked","Normalised genotype share",imgs["gen_prop"],"100%","Proportional","mix","± Shifting",3),
        mk_card(4,6,"r","Density Box Plots","Distributions","Spread per genotype",imgs["gen_box"],"3","Genotypes","flat","~ Variable",4),
        mk_card(5,6,"o","Dominance Timeline","Categorical","Leading genotype per step",imgs["gen_dom"],"—","Dominant","mix","± Switching",5),
        mk_card(6,6,"p","Pairwise Scatter Matrix","3×3 Grid","Cross-correlations A·B·C",imgs["gen_scatter"],"—","3×3 Matrix","mix","~ Correlated",6),
    ]
    mut = [
        mk_card(1,4,"y","Mutation Distribution","Histogram + KDE","Distribution of all observed rates",imgs["mut_hist"],s["mean_mutation"],"Mean Rate","flat","~ Stable",1),
        mk_card(2,4,"y","Mutation Over Time","Time Series","Rate evolution across simulation",imgs["mut_ts"],s["mean_mutation"],"Mean Rate","flat","~ Stable",2),
        mk_card(3,4,"y","Mutation Violin","Violin Plot","Full density shape",imgs["mut_vio"],"—","Full Dist.","flat","~ Stable",3),
        mk_card(4,4,"o","Mutation vs Population","Scatter","Rate vs abundance over time",imgs["mut_pop"],"—","Relationship","mix","± Linked",4),
    ]
    coop = [
        mk_card(1,5,"g","Cooperation & Competition","Dual Line","Both indices over time",imgs["cc_lines"],s["coop_comp_ratio"],"Coop/Comp Ratio","up","↑ Cooperative",1),
        mk_card(2,5,"p","Coop/Comp Ratio","Derived Metric","Above 1 = cooperation wins",imgs["cc_ratio"],s["coop_comp_ratio"],"Ratio","mix","± Variable",2),
        mk_card(3,5,"p","Phase Space Scatter","Coop vs Comp","Colour encodes time",imgs["cc_scatter"],"—","Phase Plot","mix","~ Cycling",3),
        mk_card(4,5,"g","Index Distributions","Side-by-Side Hist","Shape comparison",imgs["cc_hist"],s["mean_coop"],"Mean Coop","flat","~ Stable",4),
        mk_card(5,5,"o","Rolling Mean Comparison","Smoothed","Noise-reduced trends",imgs["cc_roll"],"—","Smoothed","mix","± Shifting",5),
    ]
    coop_extra = f"""
<div class="grid-2">
  {mk_card(6,7,"o","Cooperation vs Mutation","Scatter","Social vs evolutionary pressure",imgs["coop_mut"],"—","Relationship","mix","± Linked",6)}
  {mk_card(7,7,"r","Competition vs Mutation","Scatter","Antagonism vs mutation rate",imgs["comp_mut"],"—","Relationship","mix","± Linked",7)}
</div>"""

    cross = [
        mk_card(1,4,"t","All Variables Normalised","Overview","8 metrics on one 0–1 canvas",imgs["all_norm"],"8","Variables","mix","~ Overview",1),
        mk_card(2,4,"b","Population + Resource","Dual Axis","Two Y-axes, direct comparison",imgs["dual_ax"],"—","Dual Axis","mix","~ Coupled",2),
        mk_card(3,4,"g","Coop · Comp · Population","Normalised","Social dynamics vs abundance",imgs["ccp"],"—","3 Signals","mix","± Linked",3),
        mk_card(4,4,"r","Genotype Total vs Resource","Scatter","Carrying capacity relationship",imgs["gen_res"],"—","Scatter","mix","~ Coupled",4),
    ]

    corr_sec = f"""
<section id="correlation">
  <div class="slabel">07 — Full matrix</div>
  <h2>Pearson Correlation Heatmap</h2>
  <div class="grid-2">
    <div class="card p full d1">
      <div class="card-head">
        <div class="card-meta">
          <span class="card-num" style="color:var(--purple)">8 × 8</span>
          <span class="card-title">All Variables — Pearson Correlation</span>
          <span class="badge" style="color:var(--purple);background:rgba(199,125,255,.08);border:1px solid rgba(199,125,255,.15)">Full Heatmap</span>
        </div>
        <div class="card-note">Pairwise linear correlations across every simulation metric</div>
      </div>
      <div class="chart-body"><img src="data:image/png;base64,{imgs['corr']}" alt="Correlation Heatmap" style="max-width:520px;margin:auto"></div>
      <div class="card-foot">
        <div class="metric"><span class="mv">8 × 8</span><span class="ml">Variable Matrix</span></div>
        <span class="trend mix">± Mixed</span>
      </div>
    </div>
  </div>
</section>"""

    # ── NEW: Simulation Parameters & Environment Section ──────────────────────
    sim_env_cards = [
        mk_card(1,10,"b","Resource Regime","Environment Type","Rich vs depleted periods · median split",        imgs["env_resource"], s["rich_pct"],      "Time Resource-Rich","mix","± Alternating",1),
        mk_card(2,10,"r","Antibiotic Stress","Mutation Spikes","Sudden & gradual stress proxy from μ spikes",   imgs["antibiotic"],   s["stress_events"],  "Stress Events",     "dn","↑ Stressed",   2),
        mk_card(3,10,"p","Fitness Landscape","Evolutionary Driver","ΔPop / Resource — landscape curvature",    imgs["fitness_land"], "—",                 "Fitness Proxy",     "mix","± Variable",  3),
        mk_card(4,10,"o","Carrying Capacity","K-Saturation","Population % of running max — proximity to K",    imgs["carry_cap"],    "K-proxy",           "% Saturation",      "flat","~ Near K",   4),
        mk_card(5,10,"t","Generation Time","Doubling Rate","log₂ doublings per step — inverse gen. time",      imgs["gen_time"],     "—",                 "Doublings / Step",  "up","↑ Dividing",  5),
        mk_card(6,10,"r","Adversarial Dynamics","Competition + Depletion","Resource war & competitive exclusion pressure",imgs["adversarial"],"—","Dual Signal","dn","↓ Hostile",6),
        mk_card(7,10,"g","Cooperative Behaviors","Public Goods + Biofilm","Cooperation & Shannon genotype diversity",      imgs["coop_behav"], "—","Coop + Entropy","up","↑ Cooperative",7),
        mk_card(8,10,"t","HGT / Quorum Sensing","Density–Cooperation","High density → elevated cooperation proxy",        imgs["hgt_quorum"], "—","Phase Plot",    "mix","~ Quorum",8),
        mk_card(9,10,"o","Spatial Distribution","Pseudo-1D Heatmap","Genotype density matrix — 1D spatial proxy",         imgs["spatial"],    "1D proxy","Spatial Layout","mix","± Patchy",1),
        mk_card(10,10,"g","Evolutionary Driver Radar","Spider Chart","6-axis normalised summary of simulation drivers",    imgs["radar"],      "6 Axes","Driver Profile","mix","~ Balanced",2),
    ]
    sim_env_sec = f"""
<section id="simenv">
  <div class="slabel">08 — Simulation Parameters &amp; Environment</div>
  <h2>Simulation Parameters &amp; Environment</h2>
  <div class="grid-2">{"".join(sim_env_cards[:8])}</div>
  <div class="grid-2">{"".join(sim_env_cards[8:])}</div>
</section>"""

    nav = "".join(
        f'<a href="#{sid}">{lbl}</a>'
        for sid, lbl in [
            ("population","Population"), ("resource","Resource"),
            ("genotypes","Genotypes"),  ("mutation","Mutation"),
            ("coop","Coop & Comp"),     ("cross","Cross-var"),
            ("correlation","Correlation"), ("simenv","Sim & Env"),
        ]
    )

    total_charts = 29 + 10   # original + new

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Bacterial Evolution Report</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>

<nav>{nav}</nav>

<header>
  <div>
    <div class="tags">
      <span class="tag g">SIM-RUN-04</span>
      <span class="tag b">{s["time_steps"]} STEPS</span>
      <span class="tag r">COMPLETE</span>
      <span class="tag y">{total_charts} CHARTS</span>
      <span class="run-id">// E. COLI K-12 · CHEMOSTAT MODEL · {s["gen_date"]}</span>
    </div>
    <h1>Bacterial<br><span>Evolution</span><br>Report</h1>
  </div>
  <div class="kpis">
    <div class="kpi"><span class="val">{s["max_pop"]}</span><span class="lbl">Peak Population</span></div>
    <div class="div-h"></div>
    <div class="kpi"><span class="val">{s["mean_coop"]}</span><span class="lbl">Mean Cooperation</span></div>
    <div class="div-h"></div>
    <div class="kpi"><span class="val">{s["mean_mutation"]}</span><span class="lbl">Mean Mut. Rate</span></div>
    <div class="div-h"></div>
    <div class="kpi"><span class="val">{s["coop_comp_ratio"]}×</span><span class="lbl">Coop/Comp Ratio</span></div>
  </div>
</header>

{sec("population","01 — Population","Population Dynamics","grid-2",pop)}
{sec("resource","02 — Resource","Resource Dynamics","grid-3",res)}
{sec("genotypes","03 — Genotypes","Genotype Competition","grid-2",gen)}
{sec("mutation","04 — Mutation","Mutation Analysis","grid-2",mut)}
{sec("coop","05 — Cooperation & Competition","Social Dynamics","grid-2",coop)}
<section style="padding:0 60px 10px;position:relative;z-index:10">{coop_extra}</section>
{sec("cross","06 — Cross-variable","Multi-metric Overview","grid-2",cross)}
{corr_sec}
{sim_env_sec}

<footer>
  <div class="foot-l">
    <div class="dot"></div>
    <span class="foot-txt">{total_charts} CHARTS · 8 VARIABLES · {s["time_steps"]} TIME STEPS · ALL MODULES NOMINAL</span>
  </div>
  <span class="foot-r">BACTERIAL EVOLUTION SIM v5.0 · {s["gen_date"]}</span>
</footer>

<script>
const secs  = document.querySelectorAll('section[id]');
const links = document.querySelectorAll('nav a');
const obs = new IntersectionObserver(entries => {{
  entries.forEach(e => {{
    if(e.isIntersecting){{
      links.forEach(l => l.classList.remove('active'));
      const a = document.querySelector(`nav a[href="#${{e.target.id}}"]`);
      if(a) a.classList.add('active');
    }}
  }});
}}, {{rootMargin:'-40% 0px -55% 0px'}});
secs.forEach(s => obs.observe(s));
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(csv_path="simulation_metrics.csv", output_path="chart.html"):
    print(f"Loading '{csv_path}' ...")
    d = pd.read_csv(csv_path)

    print("Rendering 39 charts ...")
    imgs = {
        # ── original 29 ──────────────────────────────────────────────────────
        "pop_line":   chart_population_line(d),
        "pop_roll":   chart_population_rolling(d),
        "pop_gr":     chart_population_growth_rate(d),
        "pop_log":    chart_population_log(d),
        "res_line":   chart_resource_line(d),
        "res_dr":     chart_resource_depletion_rate(d),
        "res_vs_pop": chart_resource_vs_population(d),
        "gen_lines":  chart_genotypes_lines(d),
        "gen_stack":  chart_genotypes_stacked_area(d),
        "gen_prop":   chart_genotypes_proportion(d),
        "gen_box":    chart_genotypes_boxplot(d),
        "gen_dom":    chart_genotype_dominance(d),
        "gen_scatter":chart_genotypes_scatter_matrix(d),
        "mut_hist":   chart_mutation_hist(d),
        "mut_ts":     chart_mutation_over_time(d),
        "mut_vio":    chart_mutation_violin(d),
        "mut_pop":    chart_mutation_vs_population(d),
        "cc_lines":   chart_coop_comp_lines(d),
        "cc_ratio":   chart_coop_comp_ratio(d),
        "cc_scatter": chart_coop_comp_scatter(d),
        "cc_hist":    chart_coop_comp_hist(d),
        "cc_roll":    chart_rolling_coop_comp(d),
        "coop_mut":   chart_coop_vs_mutation(d),
        "comp_mut":   chart_comp_vs_mutation(d),
        "all_norm":   chart_all_series_normalised(d),
        "dual_ax":    chart_population_resource_dual_axis(d),
        "ccp":        chart_coop_comp_population(d),
        "gen_res":    chart_genotype_resource_scatter(d),
        "corr":       chart_correlation_heatmap(d),
        # ── new 10: Simulation Parameters & Environment ───────────────────────
        "env_resource": chart_environment_resource_regime(d),
        "antibiotic":   chart_antibiotic_stress_proxy(d),
        "fitness_land": chart_fitness_landscape_proxy(d),
        "carry_cap":    chart_carrying_capacity_saturation(d),
        "gen_time":     chart_generation_time_proxy(d),
        "adversarial":  chart_adversarial_dynamics(d),
        "coop_behav":   chart_cooperative_behaviors(d),
        "hgt_quorum":   chart_hgt_quorum_proxy(d),
        "spatial":      chart_spatial_genotype_heatmap(d),
        "radar":        chart_evolutionary_driver_radar(d),
    }

    stats = compute_stats(d)
    html  = build_html(imgs, stats)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved -> {output_path}  ({len(imgs)} charts embedded)")


if __name__ == "__main__":
    generate_report()

def save_png(fig, name, folder="charts"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return path