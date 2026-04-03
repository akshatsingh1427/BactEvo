
import argparse
import base64
import json
import math
import os
import sys
from dataclasses import dataclass, asdict, field
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG    = "#060a0f"
SURFACE    = "#0c1318"
SURF2      = "#111b22"
GRID_COL   = "#1a2d38"
TEXT_COL   = "#d4eaf7"
MUTED      = "#4a7a94"

PAL = dict(
    green="#00e5a0", blue="#00b8ff", red="#ff4d6d",
    yellow="#ffd166", purple="#c77dff", orange="#ff9f43",
    teal="#4cc9f0",  pink="#f72585",
)
GEN_COLORS = [PAL["red"], PAL["yellow"], PAL["green"]]
GEN_NAMES  = ["Genotype A", "Genotype B", "Genotype C"]

# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SimParams:
    # ── Environment ──────────────────────────────────────────────────────────
    env_type:            str   = "rich"   # rich | depleted | antibiotic_gradual | antibiotic_spike
    inflow:              float = 2.0      # resource inflow (mM/t)
    resource_decay:      float = 0.05
    initial_resource:    float = 100.0

    # ── Evolutionary Drivers ─────────────────────────────────────────────────
    mutation_rate:       float = 0.001
    hgt_rate:            float = 0.0005   # horizontal gene transfer
    carrying_capacity:   float = 1000.0
    generation_time:     float = 1.0
    mutation_benefit:    float = 0.02

    # ── Initial Populations ──────────────────────────────────────────────────
    init_A:              float = 200.0
    init_B:              float = 150.0
    init_C:              float = 50.0

    # ── Fitness Landscape ────────────────────────────────────────────────────
    fitness_A:           float = 1.00
    fitness_B:           float = 0.95
    fitness_C:           float = 1.10

    # ── Adversarial ──────────────────────────────────────────────────────────
    bacteriocin:         float = 0.003   # dominant suppresses others
    competition_alpha:   float = 0.80    # inter-specific competition coeff
    resource_consumption:float = 0.002

    # ── Cooperative ──────────────────────────────────────────────────────────
    public_good_benefit: float = 0.05
    quorum_threshold:    float = 300.0
    biofilm_threshold:   float = 400.0
    biofilm_protection:  float = 0.60

    # ── Antibiotic ───────────────────────────────────────────────────────────
    antibiotic_onset:    float = 30.0
    antibiotic_max:      float = 0.30
    antibiotic_ramp:     float = 20.0
    resistance_A:        float = 0.20
    resistance_B:        float = 0.50
    resistance_C:        float = 0.80

    # ── Simulation ───────────────────────────────────────────────────────────
    t_end:               float = 200.0
    dt_out:              float = 0.5      # output resolution
    spatial_size:        int   = 20       # NxN grid

# ══════════════════════════════════════════════════════════════════════════════
#  ODE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
def make_ode(p: SimParams):
    """
    State vector y = [N_A, N_B, N_C, R, mut_A, mut_B, mut_C, fit_A, fit_B, fit_C]
    Returns the derivative function for solve_ivp.
    """
    K  = p.carrying_capacity
    mu = p.mutation_rate
    base_fitness = [p.fitness_A, p.fitness_B, p.fitness_C]
    resistances  = [p.resistance_A, p.resistance_B, p.resistance_C]

    def antibiotic(t):
        if p.env_type == "antibiotic_gradual":
            if t < p.antibiotic_onset:
                return 0.0
            return min(1.0, (t - p.antibiotic_onset) / p.antibiotic_ramp)
        elif p.env_type == "antibiotic_spike":
            if t < p.antibiotic_onset:
                return 0.0
            return max(0.0, 1.0 - (t - p.antibiotic_onset) * 0.04)
        return 0.0

    def ode(t, y):
        N  = y[0:3].clip(0)          # genotype densities
        R  = max(0.0, y[3])          # resource
        mb = y[4:7]                  # mutation burdens
        fi = y[7:10]                 # evolving fitness values

        total_N = N.sum()
        ab_conc = antibiotic(t)

        # Environment modifier
        env_mult = 0.3 if p.env_type == "depleted" else 1.0

        # Quorum & biofilm
        quorum  = total_N > p.quorum_threshold
        biofilm = total_N > p.biofilm_threshold
        ab_eff  = ab_conc * (1 - p.biofilm_protection if biofilm else 1.0)

        # Dominant genotype (for bacteriocin)
        dominant = int(np.argmax(N))

        # Monod resource factor
        res_factor = R / (R + 20.0) if R > 0 else 0.0

        dN = np.zeros(3)
        for i in range(3):
            Ni = N[i]
            # Effective fitness
            eff_fi = fi[i] + (p.public_good_benefit if quorum else 0) \
                            + mb[i] * p.mutation_benefit

            # Logistic growth with inter-specific competition
            comp_sum = sum((1.0 if j == i else p.competition_alpha) * N[j]
                           for j in range(3))
            growth = (eff_fi / p.generation_time) * Ni \
                     * (1 - comp_sum / K) * res_factor

            # Mutation flux (gain from j, lose to j)
            mut_flux = sum(mu * N[j] for j in range(3) if j != i) \
                     - sum(mu * Ni   for j in range(3) if j != i)

            # HGT: receive from fitter neighbour genotypes
            hgt_flux = sum(p.hgt_rate * N[j] * Ni * 0.01
                           for j in range(3) if j != i and N[j] > Ni)

            # Bacteriocin suppression
            bact_loss = (p.bacteriocin * N[dominant] * Ni
                         if i != dominant else 0.0)

            # Antibiotic kill
            kill = ab_eff * (1 - resistances[i]) * p.antibiotic_max * Ni

            dN[i] = growth + mut_flux + hgt_flux - bact_loss - kill

        # Resource ODE
        total_consumption = p.resource_consumption * total_N
        dR = env_mult * p.inflow - total_consumption - p.resource_decay * R

        # Mutation burden accumulation
        dmb = mu * N * 0.1

        # Fitness evolution (Ornstein-Uhlenbeck drift toward base)
        theta = 0.02
        sigma = 0.001
        dfit  = theta * (np.array(base_fitness) - fi) \
              + sigma * np.random.randn(3)

        return np.concatenate([dN, [dR], dmb, dfit])

    return ode

# ══════════════════════════════════════════════════════════════════════════════
#  RUN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def run_simulation(p: SimParams) -> dict:
    y0 = np.array([
        p.init_A, p.init_B, p.init_C,   # N
        p.initial_resource,               # R
        0.0, 0.0, 0.0,                   # mutation burdens
        p.fitness_A, p.fitness_B, p.fitness_C,  # fitness
    ])

    t_eval = np.arange(0, p.t_end, p.dt_out)
    ode_fn = make_ode(p)

    sol = solve_ivp(
        ode_fn, [0, p.t_end], y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-4, atol=1e-6,
        dense_output=False,
    )

    t   = sol.t
    N   = sol.y[0:3].clip(0)
    R   = sol.y[3].clip(0)
    mb  = sol.y[4:7].clip(0)
    fit = sol.y[7:10]

    total_N = N.sum(axis=0)

    # Antibiotic profile
    ab_fn = make_ode(p).__closure__  # recompute simply
    def ab(t_):
        if p.env_type == "antibiotic_gradual":
            if t_ < p.antibiotic_onset: return 0.0
            return min(1.0, (t_ - p.antibiotic_onset) / p.antibiotic_ramp)
        elif p.env_type == "antibiotic_spike":
            if t_ < p.antibiotic_onset: return 0.0
            return max(0.0, 1.0 - (t_ - p.antibiotic_onset) * 0.04)
        return 0.0

    ab_arr = np.array([ab(ti) for ti in t])
    quorum  = (total_N > p.quorum_threshold).astype(float)
    biofilm = (total_N > p.biofilm_threshold).astype(float)
    prop    = N / np.where(total_N > 0, total_N, 1)

    # Cooperation & competition indices
    coop_idx = quorum * p.public_good_benefit
    comp_idx = p.bacteriocin * N.max(axis=0)

    return dict(
        t=t, N=N, R=R, mb=mb, fit=fit,
        total_N=total_N, ab=ab_arr,
        quorum=quorum, biofilm=biofilm,
        prop=prop,
        coop_idx=coop_idx, comp_idx=comp_idx,
        params=p,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  SPATIAL SIMULATION (2-D diffusion)
# ══════════════════════════════════════════════════════════════════════════════
def run_spatial(p: SimParams, res: dict, n_snaps: int = 8) -> list:
    """
    Run a lightweight 2-D diffusion simulation for spatial snapshots.
    Returns list of (step_label, grid[N,N,3]) tuples.
    """
    N_  = p.spatial_size
    D   = 0.08          # diffusion coefficient
    t   = res["t"]
    pops= res["N"]      # shape (3, T)
    T   = len(t)

    # Initialise grid: gaussian blob at centre
    cx = cy = N_ // 2
    grid = np.zeros((N_, N_, 3))
    for r in range(N_):
        for c in range(N_):
            dist = math.sqrt((r - cy)**2 + (c - cx)**2)
            sigma = N_ / 4
            w = math.exp(-dist**2 / (2 * sigma**2))
            grid[r, c, 0] = p.init_A * w * (0.8 + np.random.rand() * 0.4)
            grid[r, c, 1] = p.init_B * w * 0.7 * (0.8 + np.random.rand() * 0.4)
            grid[r, c, 2] = p.init_C * w * 0.3 * (0.8 + np.random.rand() * 0.4)

    snap_indices = np.linspace(0, T - 1, n_snaps, dtype=int)
    snapshots = []

    for step_idx in range(T):
        # Laplacian diffusion (periodic boundary)
        lap = (np.roll(grid, 1, 0) + np.roll(grid, -1, 0)
             + np.roll(grid, 1, 1) + np.roll(grid, -1, 1)
             - 4 * grid)
        grid += D * lap * p.dt_out
        grid = grid.clip(0)

        # Rescale each genotype to match global ODE totals
        for g in range(3):
            cell_sum = grid[:, :, g].sum()
            if cell_sum > 0:
                grid[:, :, g] *= pops[g, step_idx] / cell_sum

        if step_idx in snap_indices:
            snapshots.append((round(t[step_idx], 1), grid.copy()))

    return snapshots

# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def apply_style(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def mk_legend(ax, **kw):
    kw.setdefault("fontsize", 8)
    kw.setdefault("labelcolor", TEXT_COL)
    lg = ax.legend(frameon=False, **kw)
    return lg


def to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG)
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return enc


def fmt_k(x, _):
    if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# ══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def chart_genotype_densities(res) -> str:
    t, N = res["t"], res["N"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    apply_style(ax, fig)
    for i, (clr, lbl) in enumerate(zip(GEN_COLORS, GEN_NAMES)):
        ax.fill_between(t, N[i], alpha=0.12, color=clr)
        ax.plot(t, N[i], color=clr, lw=1.8, label=lbl)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    ax.set_xlabel("Time"); ax.set_ylabel("Density")
    mk_legend(ax)
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_total_population(res) -> str:
    t, total = res["t"], res["total_N"]
    K = res["params"].carrying_capacity
    fig, ax = plt.subplots(figsize=(6, 3.2))
    apply_style(ax, fig)
    ax.fill_between(t, total, alpha=0.13, color=PAL["green"])
    ax.plot(t, total, color=PAL["green"], lw=2, label="Total Population")
    ax.axhline(K, color=MUTED, lw=1, linestyle=":", label=f"K = {K:,.0f}")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    ax.set_xlabel("Time"); ax.set_ylabel("Population")
    mk_legend(ax)
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_proportion(res) -> str:
    t, prop = res["t"], res["prop"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    apply_style(ax, fig)
    ax.stackplot(t, prop[0], prop[1], prop[2],
                 labels=GEN_NAMES, colors=GEN_COLORS, alpha=0.78)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_xlabel("Time"); ax.set_ylabel("Proportion")
    mk_legend(ax, loc="upper left")
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_growth_rate(res) -> str:
    t, total = res["t"], res["total_N"]
    gr = np.diff(total) / np.where(total[:-1] > 0, total[:-1], 1) * 100
    t2 = t[1:]
    fig, ax = plt.subplots(figsize=(6, 3.0))
    apply_style(ax, fig)
    ax.axhline(0, color=GRID_COL, lw=1)
    ax.fill_between(t2, gr, where=gr >= 0, alpha=0.22, color=PAL["green"], interpolate=True)
    ax.fill_between(t2, gr, where=gr < 0,  alpha=0.22, color=PAL["red"],   interpolate=True)
    ax.plot(t2, gr, color=PAL["teal"], lw=1.2)
    ax.set_xlabel("Time"); ax.set_ylabel("Growth Rate (%)")
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_fitness(res) -> str:
    t, fit = res["t"], res["fit"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    apply_style(ax, fig)
    for i, (clr, lbl) in enumerate(zip(GEN_COLORS, GEN_NAMES)):
        ax.plot(t, fit[i], color=clr, lw=1.8, label=f"Fitness {lbl[-1]}")
    ax.set_xlabel("Time"); ax.set_ylabel("Fitness")
    mk_legend(ax)
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_fitness_vs_population(res) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    for i, (ax, clr, lbl) in enumerate(zip(axes, GEN_COLORS, GEN_NAMES)):
        apply_style(ax, fig)
        sc = ax.scatter(res["fit"][i], res["N"][i],
                        c=res["t"], cmap="cool", s=6, alpha=0.6, linewidths=0)
        ax.set_xlabel(f"Fitness {lbl[-1]}", fontsize=8)
        ax.set_ylabel("Density" if i == 0 else "", fontsize=8)
        ax.set_title(lbl, color=clr, fontsize=9, fontfamily="monospace")
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_fitness_bar(res) -> str:
    p = res["params"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    apply_style(ax, fig)
    names = GEN_NAMES
    final_fit = res["fit"][:, -1]
    resistances = [p.resistance_A, p.resistance_B, p.resistance_C]
    x = np.arange(3)
    w = 0.35
    bars1 = ax.bar(x - w/2, final_fit, width=w, color=GEN_COLORS,
                   alpha=0.7, label="Fitness", edgecolor="none")
    bars2 = ax.bar(x + w/2, resistances, width=w,
                   color=PAL["teal"], alpha=0.55, label="AB Resistance", edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(["Gen A", "Gen B", "Gen C"], color=MUTED, fontsize=8)
    mk_legend(ax)
    ax.set_ylabel("Value")
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_mutation_burden(res) -> str:
    t, mb = res["t"], res["mb"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    apply_style(ax, fig)
    for i, (clr, lbl) in enumerate(zip(GEN_COLORS, GEN_NAMES)):
        ax.fill_between(t, mb[i], alpha=0.14, color=clr)
        ax.plot(t, mb[i], color=clr, lw=1.8, label=f"Burden {lbl[-1]}")
    ax.set_xlabel("Time"); ax.set_ylabel("Mutation Burden")
    mk_legend(ax)
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_mutation_rate_over_time(res) -> str:
    t, mb = res["t"], res["mb"]
    dmb = np.diff(mb, axis=1)
    t2  = t[1:]
    fig, ax = plt.subplots(figsize=(6, 3.0))
    apply_style(ax, fig)
    for i, (clr, lbl) in enumerate(zip(GEN_COLORS, GEN_NAMES)):
        ax.plot(t2, dmb[i], color=clr, lw=1.4, label=f"ΔBurden {lbl[-1]}")
    ax.axhline(0, color=GRID_COL, lw=1)
    ax.set_xlabel("Time"); ax.set_ylabel("Δ Mutation Burden / Step")
    mk_legend(ax)
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_mutation_distribution(res) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    for i, (ax, clr, lbl) in enumerate(zip(axes, GEN_COLORS, GEN_NAMES)):
        apply_style(ax, fig)
        data = res["mb"][i]
        ax.hist(data, bins=30, color=clr, alpha=0.4, edgecolor="none")
        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        xs  = np.linspace(data.min(), data.max(), 200)
        ax2.plot(xs, kde(xs), color=clr, lw=2)
        ax2.set_yticks([])
        for sp in ax2.spines.values(): sp.set_edgecolor(GRID_COL)
        ax.set_title(lbl, color=clr, fontsize=9, fontfamily="monospace")
        ax.set_xlabel("Mutation Burden", fontsize=8)
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_resource(res) -> str:
    t, R, ab = res["t"], res["R"], res["ab"]
    fig, ax1 = plt.subplots(figsize=(8, 3.2))
    apply_style(ax1, fig)
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID_COL)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax1.fill_between(t, R, alpha=0.14, color=PAL["blue"])
    ax1.plot(t, R, color=PAL["blue"], lw=2, label="Resource (mM)")
    ax2.fill_between(t, ab, alpha=0.12, color=PAL["red"])
    ax2.plot(t, ab, color=PAL["red"], lw=1.8, linestyle="--", label="Antibiotic")
    ax1.set_xlabel("Time"); ax1.set_ylabel("Resource (mM)", color=PAL["blue"])
    ax2.set_ylabel("Antibiotic Conc.", color=PAL["red"])
    ax1.tick_params(axis="y", colors=PAL["blue"])
    ax2.tick_params(axis="y", colors=PAL["red"])
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, frameon=False,
               fontsize=8, labelcolor=TEXT_COL, loc="upper right")
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_resource_vs_population(res) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    apply_style(ax, fig)
    sc = ax.scatter(res["R"], res["total_N"], c=res["t"],
                    cmap="cool", s=8, alpha=0.65, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Time", color=MUTED, fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    ax.set_xlabel("Resource (mM)"); ax.set_ylabel("Total Population")
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_antibiotic_profile(res) -> str:
    t, ab = res["t"], res["ab"]
    p = res["params"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    apply_style(ax, fig)
    # Effective kill per genotype
    for i, (clr, lbl, res_val) in enumerate(zip(
        GEN_COLORS, GEN_NAMES,
        [p.resistance_A, p.resistance_B, p.resistance_C]
    )):
        eff_kill = ab * (1 - res_val) * p.antibiotic_max
        ax.plot(t, eff_kill, color=clr, lw=1.5, label=f"Effective kill — {lbl[-1]}")
    ax.fill_between(t, ab * p.antibiotic_max, alpha=0.08, color=PAL["red"])
    ax.plot(t, ab * p.antibiotic_max, color=PAL["red"], lw=2, linestyle=":",
            label="Max kill (no resistance)")
    ax.set_xlabel("Time"); ax.set_ylabel("Kill Rate")
    mk_legend(ax, loc="upper left")
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_social(res) -> str:
    t = res["t"]
    coop = res["coop_idx"]
    comp = res["comp_idx"]
    ratio = np.where(comp > 0, coop / comp, 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    # Left: coop + comp over time
    ax = axes[0]; apply_style(ax, fig)
    ax.fill_between(t, coop, alpha=0.14, color=PAL["green"])
    ax.fill_between(t, comp, alpha=0.14, color=PAL["red"])
    ax.plot(t, coop, color=PAL["green"], lw=2, label="Cooperation Index")
    ax.plot(t, comp, color=PAL["red"],   lw=2, label="Competition Index")
    ax.set_xlabel("Time"); ax.set_ylabel("Index")
    mk_legend(ax)
    # Right: ratio
    ax = axes[1]; apply_style(ax, fig)
    ax.axhline(1.0, color=MUTED, lw=1, linestyle=":", label="Balanced (ratio=1)")
    ax.fill_between(t, ratio, 1, where=ratio >= 1, alpha=0.18,
                    color=PAL["green"], interpolate=True, label="Coop dominant")
    ax.fill_between(t, ratio, 1, where=ratio < 1,  alpha=0.18,
                    color=PAL["red"],   interpolate=True, label="Comp dominant")
    ax.plot(t, ratio, color=PAL["purple"], lw=2)
    ax.set_xlabel("Time"); ax.set_ylabel("Coop / Comp Ratio")
    mk_legend(ax)
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_quorum_biofilm(res) -> str:
    t = res["t"]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    apply_style(ax, fig)
    ax.fill_between(t, res["quorum"], step="post", alpha=0.30,
                    color=PAL["purple"], label="Quorum Active")
    ax.fill_between(t, res["biofilm"], step="post", alpha=0.22,
                    color=PAL["orange"], label="Biofilm Active")
    ax.step(t, res["quorum"],  color=PAL["purple"], lw=2, where="post")
    ax.step(t, res["biofilm"], color=PAL["orange"], lw=2, where="post", linestyle="--")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["OFF", "ON"])
    ax.set_xlabel("Time"); ax.set_ylabel("Status")
    mk_legend(ax, loc="upper right")
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_correlation_heatmap(res) -> str:
    cols = {
        "Pop A": res["N"][0], "Pop B": res["N"][1], "Pop C": res["N"][2],
        "Resource": res["R"], "Antibiotic": res["ab"],
        "Fitness A": res["fit"][0], "Fit B": res["fit"][1], "Fit C": res["fit"][2],
        "Mut A": res["mb"][0], "Mut B": res["mb"][1],
        "Coop": res["coop_idx"], "Comp": res["comp_idx"],
    }
    import pandas as pd
    df = pd.DataFrame(cols)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    apply_style(ax, fig)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", annot_kws={"size": 7},
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.4, linecolor=DARK_BG,
                cbar_kws={"shrink": 0.72})
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color=TEXT_COL)
    ax.set_yticklabels(ax.get_yticklabels(), color=TEXT_COL)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_spatial_panels(snapshots: list) -> str:
    n = len(snapshots)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.2, 6.5))
    if n == 1: axes = axes.reshape(3, 1)
    fig.patch.set_facecolor(DARK_BG)
    cmaps = ["Reds", "YlOrBr", "Greens"]
    for col, (label, grid) in enumerate(snapshots):
        for row in range(3):
            ax = axes[row, col]
            data = grid[:, :, row]
            ax.imshow(data, cmap=cmaps[row], aspect="auto",
                      interpolation="bilinear",
                      vmin=0, vmax=max(data.max(), 1e-9))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_COL)
            if col == 0:
                ax.set_ylabel(GEN_NAMES[row], color=GEN_COLORS[row],
                               fontsize=8, fontfamily="monospace")
            if row == 0:
                ax.set_title(f"t={label}", color=MUTED, fontsize=8,
                             fontfamily="monospace")
    fig.tight_layout(pad=0.8)
    return to_b64(fig)


def chart_dominance_timeline(res) -> str:
    t, N = res["t"], res["N"]
    dominant = np.argmax(N, axis=0)
    fig, ax = plt.subplots(figsize=(8, 1.5))
    apply_style(ax, fig)
    from matplotlib.collections import LineCollection
    points = np.array([t, np.zeros_like(t)]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, colors=[GEN_COLORS[d] for d in dominant[:-1]], lw=6)
    ax.add_collection(lc)
    ax.set_xlim(t.min(), t.max()); ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(GEN_COLORS, GEN_NAMES)]
    ax.legend(handles=patches, frameon=False, fontsize=8,
              labelcolor=TEXT_COL, loc="upper right", ncol=3)
    fig.tight_layout(pad=1.0)
    return to_b64(fig)


def chart_phase_portrait(res) -> str:
    """Phase portrait: N_A vs N_B, coloured by N_C density."""
    fig, ax = plt.subplots(figsize=(6, 4))
    apply_style(ax, fig)
    sc = ax.scatter(res["N"][0], res["N"][1], c=res["N"][2],
                    cmap="summer", s=6, alpha=0.7, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Genotype C density", color=MUTED, fontsize=8)
    ax.set_xlabel("Genotype A density"); ax.set_ylabel("Genotype B density")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_k))
    # Add trajectory arrow
    mid = len(res["t"]) // 2
    ax.annotate("", xy=(res["N"][0][mid+2], res["N"][1][mid+2]),
                xytext=(res["N"][0][mid], res["N"][1][mid]),
                arrowprops=dict(arrowstyle="->", color=PAL["teal"], lw=1.5))
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


def chart_all_normalised(res) -> str:
    t = res["t"]
    series = [
        ("Total Pop",    res["total_N"],    PAL["green"]),
        ("Resource",     res["R"],          PAL["blue"]),
        ("Gen A",        res["N"][0],       PAL["red"]),
        ("Gen B",        res["N"][1],       PAL["yellow"]),
        ("Gen C",        res["N"][2],       PAL["green"]),
        ("Fitness A",    res["fit"][0],     PAL["orange"]),
        ("Mutation A",   res["mb"][0],      PAL["purple"]),
        ("Cooperation",  res["coop_idx"],   PAL["teal"]),
        ("Competition",  res["comp_idx"],   PAL["pink"]),
        ("Antibiotic",   res["ab"],         PAL["red"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    apply_style(ax, fig)
    for lbl, data, clr in series:
        rng = data.max() - data.min()
        norm = (data - data.min()) / rng if rng > 0 else data * 0
        ax.plot(t, norm, label=lbl, color=clr, lw=1.1, alpha=0.82)
    ax.set_xlabel("Time"); ax.set_ylabel("Normalised (0–1)")
    mk_legend(ax, loc="upper right", ncol=2, fontsize=7)
    fig.tight_layout(pad=1.2)
    return to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY STATS
# ══════════════════════════════════════════════════════════════════════════════
def compute_summary(res: dict, p: SimParams) -> dict:
    N = res["N"]; total = res["total_N"]
    survived = [N[i][-1] > 5 for i in range(3)]
    peak_idx  = np.argmax(total)
    mean_coop = res["coop_idx"].mean()
    mean_comp = res["comp_idx"].mean()
    return dict(
        max_total    = f"{int(total.max()):,}",
        peak_time    = f"{res['t'][peak_idx]:.1f}",
        final_A      = f"{int(N[0][-1]):,}",
        final_B      = f"{int(N[1][-1]):,}",
        final_C      = f"{int(N[2][-1]):,}",
        survived     = survived,
        min_resource = f"{res['R'].min():.2f} mM",
        max_ab       = f"{res['ab'].max()*100:.0f}%",
        quorum_pct   = f"{res['quorum'].mean()*100:.0f}%",
        biofilm_pct  = f"{res['biofilm'].mean()*100:.0f}%",
        mean_coop    = f"{mean_coop:.4f}",
        mean_comp    = f"{mean_comp:.4f}",
        coop_comp_ratio = f"{(mean_coop/mean_comp):.2f}" if mean_comp > 0 else "∞",
        n_steps      = len(res["t"]),
        env_type     = p.env_type.replace("_", " ").title(),
        gen_date     = datetime.now().strftime("%Y-%m-%d %H:%M"),
        mutation_rate= p.mutation_rate,
        hgt_rate     = p.hgt_rate,
        K            = f"{p.carrying_capacity:,.0f}",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  JAVASCRIPT SIMULATION ENGINE (embedded in HTML)
#  Mirrors the Python ODE in pure JS so sliders instantly re-run in browser.
# ══════════════════════════════════════════════════════════════════════════════
JS_ENGINE = r"""
// ── Lightweight JS simulation (Euler method) ──────────────────────────────
function runJS(p) {
  const K   = p.carryingCapacity;
  const mu  = p.mutationRate;
  const dt  = 0.5;
  const steps = Math.round(p.tEnd / dt);

  let N   = [p.initA, p.initB, p.initC];
  let R   = p.initialResource;
  let mb  = [0, 0, 0];
  let fit = [p.fitnessA, p.fitnessB, p.fitnessC];
  const resistances = [p.resistanceA, p.resistanceB, p.resistanceC];
  const base_fit    = [p.fitnessA, p.fitnessB, p.fitnessC];

  const hist = [];

  for (let step = 0; step < steps; step++) {
    const t = step * dt;

    // Antibiotic
    let abConc = 0;
    if (p.envType === "antibiotic_gradual" && t >= p.antibioticOnset) {
      abConc = Math.min(1, (t - p.antibioticOnset) / p.antibioticRamp);
    } else if (p.envType === "antibiotic_spike" && t >= p.antibioticOnset) {
      abConc = Math.max(0, 1 - (t - p.antibioticOnset) * 0.04);
    }

    const envMult = p.envType === "depleted" ? 0.3 : 1.0;
    const totalN  = N.reduce((a, b) => a + b, 0);
    const quorum  = totalN > p.quorumThreshold;
    const biofilm = totalN > p.biofilmThreshold;
    const abEff   = abConc * (biofilm ? (1 - p.biofilmProtection) : 1);
    const resFac  = R > 0 ? R / (R + 20) : 0;
    const dom     = N.indexOf(Math.max(...N));

    const dN = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      const Ni = Math.max(N[i], 0);
      const effFit = fit[i]
        + (quorum ? p.publicGoodBenefit : 0)
        + mb[i] * p.mutationBenefit;
      const compSum = N.reduce((s, Nj, j) => s + (i === j ? 1 : p.competitionAlpha) * Nj, 0);
      const growth  = (effFit / p.generationTime) * Ni * (1 - compSum / K) * resFac;
      const mutFlux = N.reduce((s, Nj, j) => j !== i ? s + mu * (Nj - Ni) : s, 0);
      const hgtFlux = N.reduce((s, Nj, j) => j !== i && Nj > Ni ? s + p.hgtRate * Nj * Ni * 0.01 : s, 0);
      const bact    = i !== dom ? p.bacteriocin * N[dom] * Ni : 0;
      const kill    = abEff * (1 - resistances[i]) * p.antibioticMax * Ni;
      dN[i] = (growth + mutFlux + hgtFlux - bact - kill) * dt;
    }

    const totalConsume = p.resourceConsumption * totalN;
    const dR = (envMult * p.inflow - totalConsume - p.resourceDecay * R) * dt;

    N  = N.map((v, i) => Math.max(0.1, v + dN[i]));
    R  = Math.max(0, R + dR);
    mb = mb.map((v, i) => v + mu * N[i] * 0.1 * dt);
    fit = fit.map((v, i) => v + 0.02 * (base_fit[i] - v) * dt);

    const tot = N.reduce((a, b) => a + b, 0);
    const coop = quorum ? p.publicGoodBenefit : 0;
    const comp = p.bacteriocin * Math.max(...N);

    hist.push({
      t: Math.round(t * 10) / 10,
      pA: Math.round(N[0]), pB: Math.round(N[1]), pC: Math.round(N[2]),
      tot: Math.round(tot),
      res: Math.round(R * 10) / 10,
      ab: Math.round(abEff * 1000) / 1000,
      fA: Math.round(fit[0]*1000)/1000,
      fB: Math.round(fit[1]*1000)/1000,
      fC: Math.round(fit[2]*1000)/1000,
      mA: Math.round(mb[0]*100)/100,
      mB: Math.round(mb[1]*100)/100,
      mC: Math.round(mb[2]*100)/100,
      quorum: quorum ? 1 : 0,
      biofilm: biofilm ? 1 : 0,
      coop, comp,
    });
  }
  return hist;
}

// ── Canvas chart helpers ───────────────────────────────────────────────────
const BG = "#060a0f", SURF = "#0c1318", GRID = "#1a2d38", MUT = "#4a7a94";
const GC = ["#ff4d6d","#ffd166","#00e5a0"];
const GN = ["Genotype A","Genotype B","Genotype C"];

function clearCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = SURF;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  return ctx;
}

function drawLineChart(canvas, datasets, opts = {}) {
  const ctx = clearCanvas(canvas);
  const W = canvas.width, H = canvas.height;
  const PAD = { l: 52, r: 16, t: 12, b: 36 };
  const cw = W - PAD.l - PAD.r, ch = H - PAD.t - PAD.b;

  // Calculate bounds
  let yMin = opts.yMin !== undefined ? opts.yMin : Infinity;
  let yMax = opts.yMax !== undefined ? opts.yMax : -Infinity;
  let xMax = -Infinity;
  datasets.forEach(ds => {
    ds.data.forEach(d => {
      if (opts.yMin === undefined && d.y < yMin) yMin = d.y;
      if (opts.yMax === undefined && d.y > yMax) yMax = d.y;
      if (d.x > xMax) xMax = d.x;
    });
  });
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const xMin = 0;

  const tx = x => PAD.l + (x - xMin) / (xMax - xMin) * cw;
  const ty = y => PAD.t + ch - (y - yMin) / (yMax - yMin) * ch;

  // Grid lines
  ctx.strokeStyle = GRID; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const yv = yMin + i * (yMax - yMin) / 4;
    const y  = ty(yv);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + cw, y); ctx.stroke();
    ctx.fillStyle = MUT; ctx.font = "9px 'Space Mono',monospace";
    ctx.textAlign = "right";
    const label = yv >= 1e6 ? `${(yv/1e6).toFixed(1)}M`
                : yv >= 1e3 ? `${(yv/1e3).toFixed(0)}K`
                : yv.toFixed(yv < 10 ? 2 : 0);
    ctx.fillText(label, PAD.l - 4, y + 3);
  }
  for (let i = 0; i <= 5; i++) {
    const xv = xMin + i * (xMax - xMin) / 5;
    const x  = tx(xv);
    ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + ch); ctx.stroke();
    ctx.fillStyle = MUT; ctx.font = "9px 'Space Mono',monospace";
    ctx.textAlign = "center";
    ctx.fillText(xv.toFixed(0), x, PAD.t + ch + 14);
  }

  // Axes labels
  if (opts.xlabel) {
    ctx.fillStyle = MUT; ctx.font = "9px 'Space Mono',monospace";
    ctx.textAlign = "center";
    ctx.fillText(opts.xlabel, PAD.l + cw / 2, H - 4);
  }
  if (opts.ylabel) {
    ctx.save(); ctx.translate(12, PAD.t + ch / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = MUT; ctx.font = "9px 'Space Mono',monospace";
    ctx.textAlign = "center"; ctx.fillText(opts.ylabel, 0, 0);
    ctx.restore();
  }

  // Optional reference line (carrying capacity)
  if (opts.refY !== undefined) {
    const ry = ty(opts.refY);
    ctx.strokeStyle = MUT; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(PAD.l, ry); ctx.lineTo(PAD.l + cw, ry); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Lines + fills
  datasets.forEach(ds => {
    const pts = ds.data;
    if (!pts.length) return;

    // Fill
    ctx.beginPath();
    ctx.moveTo(tx(pts[0].x), ty(0));
    pts.forEach(p => ctx.lineTo(tx(p.x), ty(p.y)));
    ctx.lineTo(tx(pts[pts.length-1].x), ty(0));
    ctx.closePath();
    ctx.fillStyle = ds.color + "22"; ctx.fill();

    // Line
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(tx(p.x), ty(p.y)) : ctx.lineTo(tx(p.x), ty(p.y)));
    ctx.strokeStyle = ds.color; ctx.lineWidth = 2; ctx.stroke();
  });

  // Legend
  let lx = PAD.l + 8;
  datasets.forEach(ds => {
    ctx.fillStyle = ds.color; ctx.fillRect(lx, PAD.t + 4, 14, 3);
    ctx.fillStyle = "#d4eaf7"; ctx.font = "8px 'Space Mono',monospace";
    ctx.textAlign = "left"; ctx.fillText(ds.label, lx + 18, PAD.t + 9);
    lx += ctx.measureText(ds.label).width + 34;
  });
}

// ── Build data → chart datasets ────────────────────────────────────────────
function buildCharts(hist, params) {
  // Population chart
  drawLineChart(document.getElementById("cv-pop"), [
    { label: "Genotype A", color: GC[0], data: hist.map(d => ({x: d.t, y: d.pA})) },
    { label: "Genotype B", color: GC[1], data: hist.map(d => ({x: d.t, y: d.pB})) },
    { label: "Genotype C", color: GC[2], data: hist.map(d => ({x: d.t, y: d.pC})) },
  ], { xlabel: "Time", ylabel: "Density",
       refY: params.carryingCapacity });

  // Total population
  drawLineChart(document.getElementById("cv-total"), [
    { label: "Total Pop", color: "#00e5a0", data: hist.map(d => ({x: d.t, y: d.tot})) },
  ], { xlabel: "Time", ylabel: "Total Population",
       refY: params.carryingCapacity });

  // Resource + antibiotic
  const resMax = Math.max(...hist.map(d => d.res));
  drawLineChart(document.getElementById("cv-res"), [
    { label: "Resource (mM)", color: "#00b8ff", data: hist.map(d => ({x: d.t, y: d.res})) },
    { label: "Antibiotic ×100", color: "#ff4d6d", data: hist.map(d => ({x: d.t, y: d.ab * 100})) },
  ], { xlabel: "Time", ylabel: "Conc." });

  // Fitness
  drawLineChart(document.getElementById("cv-fit"), [
    { label: "Fitness A", color: GC[0], data: hist.map(d => ({x: d.t, y: d.fA})) },
    { label: "Fitness B", color: GC[1], data: hist.map(d => ({x: d.t, y: d.fB})) },
    { label: "Fitness C", color: GC[2], data: hist.map(d => ({x: d.t, y: d.fC})) },
  ], { xlabel: "Time", ylabel: "Fitness" });

  // Mutation burden
  drawLineChart(document.getElementById("cv-mut"), [
    { label: "Burden A", color: GC[0], data: hist.map(d => ({x: d.t, y: d.mA})) },
    { label: "Burden B", color: GC[1], data: hist.map(d => ({x: d.t, y: d.mB})) },
    { label: "Burden C", color: GC[2], data: hist.map(d => ({x: d.t, y: d.mC})) },
  ], { xlabel: "Time", ylabel: "Mutation Burden" });

  // Coop / Comp
  drawLineChart(document.getElementById("cv-social"), [
    { label: "Cooperation", color: "#00e5a0", data: hist.map(d => ({x: d.t, y: d.coop})) },
    { label: "Competition", color: "#ff4d6d", data: hist.map(d => ({x: d.t, y: d.comp})) },
  ], { xlabel: "Time", ylabel: "Index" });

  // Update KPI pills
  const last = hist[hist.length - 1];
  document.getElementById("kpi-total").textContent  = last.tot.toLocaleString();
  document.getElementById("kpi-res").textContent    = last.res + " mM";
  document.getElementById("kpi-ab").textContent     = (last.ab * 100).toFixed(0) + "%";
  document.getElementById("kpi-quorum").textContent = last.quorum ? "ACTIVE" : "—";
  document.getElementById("kpi-biofilm").textContent= last.biofilm ? "ACTIVE" : "—";
  const extA = last.pA < 5, extB = last.pB < 5, extC = last.pC < 5;
  const extCount = [extA, extB, extC].filter(Boolean).length;
  document.getElementById("kpi-extinct").textContent = extCount + "/3";
  document.getElementById("kpi-extinct").style.color = extCount > 0 ? "#ff4d6d" : "#00e5a0";
}

// ── Main simulation trigger ────────────────────────────────────────────────
function readParams() {
  const g = id => parseFloat(document.getElementById(id).value);
  const s = id => document.getElementById(id).value;
  return {
    envType: s("p-env"),
    inflow: g("p-inflow"),
    resourceDecay: g("p-rdecay"),
    initialResource: g("p-res0"),
    mutationRate: g("p-mu"),
    hgtRate: g("p-hgt"),
    carryingCapacity: g("p-K"),
    generationTime: g("p-gentime"),
    mutationBenefit: g("p-mubenefit"),
    initA: g("p-initA"), initB: g("p-initB"), initC: g("p-initC"),
    fitnessA: g("p-fA"), fitnessB: g("p-fB"), fitnessC: g("p-fC"),
    bacteriocin: g("p-bact"),
    competitionAlpha: g("p-alpha"),
    resourceConsumption: g("p-rcons"),
    publicGoodBenefit: g("p-pg"),
    quorumThreshold: g("p-qt"),
    biofilmThreshold: g("p-bt"),
    biofilmProtection: g("p-bp"),
    antibioticOnset: g("p-ab-onset"),
    antibioticMax: g("p-ab-max"),
    antibioticRamp: g("p-ab-ramp"),
    resistanceA: g("p-resA"), resistanceB: g("p-resB"), resistanceC: g("p-resC"),
    tEnd: g("p-tend"),
  };
}

let _debounce;
function rerun() {
  clearTimeout(_debounce);
  _debounce = setTimeout(() => {
    const p = readParams();
    // Update displayed slider values
    document.querySelectorAll("[data-slider]").forEach(el => {
      const valEl = document.getElementById("v-" + el.id);
      if (valEl) valEl.textContent = el.value;
    });
    const hist = runJS(p);
    buildCharts(hist, p);
  }, 80);
}

// Tab switching
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.style.display = "none");
    btn.classList.add("active");
    document.getElementById("panel-" + btn.dataset.tab).style.display = "block";
  });
});

// Initialise
window.addEventListener("load", () => { rerun(); });
document.querySelectorAll("input[type=range], select").forEach(el => {
  el.addEventListener("input", rerun);
});
"""

# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDER
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
:root{
  --bg:#060a0f;--surface:#0c1318;--surf2:#111b22;--surf3:#162430;
  --border:#1a2d38;--bh:#2a4a60;
  --green:#00e5a0;--blue:#00b8ff;--red:#ff4d6d;
  --yellow:#ffd166;--purple:#c77dff;--orange:#ff9f43;--teal:#4cc9f0;
  --text:#d4eaf7;--muted:#4a7a94;--ml:#6a9ab4;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--mono);
     background-image:linear-gradient(rgba(0,229,160,.018) 1px,transparent 1px),
                      linear-gradient(90deg,rgba(0,229,160,.018) 1px,transparent 1px);
     background-size:44px 44px; overflow:hidden; height:100vh; display:flex; flex-direction:column;}

/* TOP BAR */
#topbar{
  display:flex;align-items:center;gap:0;
  padding:0 20px;border-bottom:1px solid var(--border);
  background:rgba(6,10,15,.92);backdrop-filter:blur(12px);
  flex-shrink:0; height:54px;
}
.brand{padding:0 18px 0 0;border-right:1px solid var(--border)}
.brand-name{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:#fff;letter-spacing:-.02em}
.brand-name span{color:var(--green)}
.brand-sub{font-size:8px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-top:2px}
.kpis{display:flex;flex:1}
.kpi{display:flex;flex-direction:column;align-items:center;padding:0 14px;border-right:1px solid var(--border)}
.kpi-lbl{font-size:8px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}
.kpi-val{font-size:12px;font-weight:700;color:#fff;margin-top:1px}
.run-btn{
  background:rgba(0,229,160,.12);border:1px solid rgba(0,229,160,.3);
  border-radius:3px;color:var(--green);
  font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.14em;text-transform:uppercase;
  padding:7px 16px;cursor:pointer;transition:all .2s;margin-left:14px;
}
.run-btn:hover{background:rgba(0,229,160,.22)}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--green);margin-right:6px;animation:pulse 2s infinite}

/* MAIN LAYOUT */
#layout{display:flex;flex:1;overflow:hidden}

/* SIDEBAR */
#sidebar{
  width:260px;min-width:260px;background:var(--surface);
  border-right:1px solid var(--border);overflow-y:auto;
  transition:width .3s;flex-shrink:0;
}
#sidebar.collapsed{width:28px;min-width:28px;overflow:hidden}
.collapse-btn{
  width:100%;padding:8px 0;background:var(--surf2);border:none;
  border-bottom:1px solid var(--border);color:var(--muted);
  cursor:pointer;font-size:13px;text-align:center;flex-shrink:0;
}
.sidebar-inner{padding:14px 12px;display:block}
#sidebar.collapsed .sidebar-inner{display:none}

.sec-hdr{
  display:flex;align-items:center;gap:8px;margin:14px 0 10px;
  padding-bottom:7px;border-bottom:1px solid var(--border);
}
.sec-hdr-bar{width:3px;height:12px;border-radius:2px;flex-shrink:0}
.sec-hdr-lbl{font-size:8px;letter-spacing:.16em;text-transform:uppercase;font-weight:700}

.param{margin-bottom:12px}
.param-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px}
.param-lbl{font-size:9px;color:var(--ml);letter-spacing:.07em;text-transform:uppercase}
.param-val{font-size:10px;color:var(--text);font-weight:700}
input[type=range]{width:100%;height:4px;border-radius:2px;background:var(--border);
                  outline:none;cursor:pointer;-webkit-appearance:none;appearance:none}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:13px;height:13px;border-radius:50%;
  background:var(--green);cursor:pointer;border:2px solid var(--bg);
  box-shadow:0 0 5px rgba(0,229,160,.4);transition:box-shadow .2s;
}
input[type=range]::-webkit-slider-thumb:hover{box-shadow:0 0 9px rgba(0,229,160,.7)}
.param-desc{font-size:8px;color:var(--muted);margin-top:2px;line-height:1.4}

.btn-group{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}
.btn-opt{
  font-size:8px;letter-spacing:.1em;text-transform:uppercase;
  padding:3px 8px;border-radius:2px;cursor:pointer;
  border:1px solid var(--border);background:transparent;color:var(--muted);
  font-family:'Space Mono',monospace;transition:all .2s;
}
.btn-opt.active{border-color:var(--green);background:rgba(0,229,160,.1);color:var(--green)}

select{
  width:100%;background:var(--surf3);border:1px solid var(--border);
  border-radius:3px;color:var(--text);font-family:'Space Mono',monospace;
  font-size:10px;padding:5px 8px;outline:none;cursor:pointer;margin-bottom:8px;
}

/* CONTENT AREA */
#content{flex:1;overflow:auto;padding:16px}

/* TABS */
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:16px}
.tab-btn{
  padding:8px 14px;background:transparent;border:none;
  border-bottom:2px solid transparent;color:var(--muted);
  font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.12em;
  text-transform:uppercase;cursor:pointer;transition:all .2s;margin-bottom:-1px;
}
.tab-btn.active{color:var(--green);border-bottom-color:var(--green)}
.tab-btn:hover:not(.active){color:var(--text)}

/* CHART CARDS */
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
.full{grid-column:1/-1}

.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:4px;overflow:hidden;transition:border-color .25s;
}
.card:hover{border-color:rgba(0,229,160,.2)}
.card-hdr{
  padding:10px 14px 8px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;
  background:var(--surf2);border-top:2px solid;
}
.card-title{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;color:#fff}
.badge{
  font-size:7px;letter-spacing:.12em;text-transform:uppercase;
  padding:2px 6px;border-radius:2px;
}
.card-body{padding:12px 10px;background:var(--surf2)}
.card-body img{width:100%;height:auto;display:block;border-radius:2px}
canvas{display:block;border-radius:2px;background:var(--surface)}

/* STAT GRID */
.stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.stat-box{
  background:var(--surf3);border-radius:3px;padding:8px 10px;
  border:1px solid var(--border);
}
.stat-box-lbl{font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.stat-box-val{font-size:13px;font-weight:700;color:#fff;margin-top:3px}
.stat-box-sub{font-size:8px;margin-top:2px}

/* Scrollbar */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,229,160,.4)}50%{opacity:.6;box-shadow:0 0 0 5px rgba(0,229,160,0)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
"""


def param_slider(pid, label, value, mn, mx, step=0.01, desc="", color="var(--green)"):
    return f"""
<div class="param">
  <div class="param-row">
    <span class="param-lbl">{label}</span>
    <span class="param-val" id="v-{pid}">{value}</span>
  </div>
  <input type="range" id="{pid}" data-slider="1"
         min="{mn}" max="{mx}" step="{step}" value="{value}"
         style="accent-color:{color}">
  {'<div class="param-desc">' + desc + '</div>' if desc else ''}
</div>"""


def sec_hdr(icon, label, color):
    return f"""
<div class="sec-hdr">
  <div class="sec-hdr-bar" style="background:{color}"></div>
  <span class="sec-hdr-lbl" style="color:{color}">{icon} {label}</span>
</div>"""


def card(title, badge, color, content_html, span=1, canvas_id=None):
    cls = "card full" if span > 1 else "card"
    body = f'<canvas id="{canvas_id}" style="width:100%;height:200px"></canvas>' if canvas_id else content_html
    return f"""
<div class="{cls}">
  <div class="card-hdr" style="border-top-color:{color}">
    <span class="card-title">{title}</span>
    <span class="badge" style="color:{color};background:{color}18;border:1px solid {color}30">{badge}</span>
  </div>
  <div class="card-body">{('<canvas id="' + canvas_id + '" width="700" height="200" style="width:100%;height:auto"></canvas>') if canvas_id else content_html}</div>
</div>"""


def build_html(imgs: dict, summary: dict, p: SimParams) -> str:
    PA = asdict(p)

    # Build sidebar sliders
    sidebar_html = f"""
<button class="collapse-btn" onclick="
  const sb=document.getElementById('sidebar');
  sb.classList.toggle('collapsed');
  this.textContent=sb.classList.contains('collapsed')?'›':'‹';
">‹</button>
<div class="sidebar-inner">

{sec_hdr("🌍","Environment",PAL["blue"])}
<div class="param">
  <div class="param-lbl" style="margin-bottom:5px">Environment Type</div>
  <select id="p-env" onchange="rerun()">
    <option value="rich" {"selected" if p.env_type=="rich" else ""}>Rich</option>
    <option value="depleted" {"selected" if p.env_type=="depleted" else ""}>Depleted</option>
    <option value="antibiotic_gradual" {"selected" if p.env_type=="antibiotic_gradual" else ""}>Antibiotic — Gradual</option>
    <option value="antibiotic_spike" {"selected" if p.env_type=="antibiotic_spike" else ""}>Antibiotic — Spike</option>
  </select>
</div>
{param_slider("p-inflow","Resource Inflow",p.inflow,0,10,0.1,"mM per time step",PAL["blue"])}
{param_slider("p-rdecay","Resource Decay",p.resource_decay,0,0.5,0.005,"",PAL["blue"])}
{param_slider("p-res0","Initial Resource",p.initial_resource,10,500,5,"mM",PAL["blue"])}

{sec_hdr("🧬","Evolutionary Drivers",PAL["green"])}
{param_slider("p-mu","Mutation Rate",p.mutation_rate,0,0.02,0.0001,"Per-capita per step")}
{param_slider("p-hgt","HGT Rate",p.hgt_rate,0,0.005,0.0001,"Horizontal gene transfer")}
{param_slider("p-K","Carrying Capacity",int(p.carrying_capacity),100,5000,50,"")}
{param_slider("p-gentime","Generation Time",p.generation_time,0.1,5,0.1,"Scales growth rate")}
{param_slider("p-mubenefit","Mutation Benefit",p.mutation_benefit,0,0.2,0.005,"")}

{sec_hdr("📈","Fitness Landscape",PAL["yellow"])}
{param_slider("p-fA","Fitness A",p.fitness_A,0.1,2,0.01,"",PAL["red"])}
{param_slider("p-fB","Fitness B",p.fitness_B,0.1,2,0.01,"",PAL["yellow"])}
{param_slider("p-fC","Fitness C",p.fitness_C,0.1,2,0.01,"",PAL["green"])}

{sec_hdr("🔬","Initial Populations",PAL["teal"])}
{param_slider("p-initA","Init A",int(p.init_A),1,1000,10,"",PAL["red"])}
{param_slider("p-initB","Init B",int(p.init_B),1,1000,10,"",PAL["yellow"])}
{param_slider("p-initC","Init C",int(p.init_C),1,1000,10,"",PAL["green"])}

{sec_hdr("⚔️","Adversarial",PAL["red"])}
{param_slider("p-bact","Bacteriocin",p.bacteriocin,0,0.02,0.0001,"Dominant suppresses others",PAL["red"])}
{param_slider("p-alpha","Competition α",p.competition_alpha,0,2,0.05,"Inter-specific coeff",PAL["red"])}
{param_slider("p-rcons","Resource Use",p.resource_consumption,0,0.01,0.0001,"Per-capita",PAL["red"])}

{sec_hdr("🤝","Cooperative",PAL["purple"])}
{param_slider("p-pg","Public Good",p.public_good_benefit,0,0.3,0.005,"Fitness bonus at quorum",PAL["purple"])}
{param_slider("p-qt","Quorum Threshold",int(p.quorum_threshold),50,2000,25,"",PAL["purple"])}
{param_slider("p-bt","Biofilm Threshold",int(p.biofilm_threshold),100,3000,50,"",PAL["orange"])}
{param_slider("p-bp","Biofilm Protection",p.biofilm_protection,0,1,0.01,"Reduces AB efficacy",PAL["orange"])}

{sec_hdr("💊","Antibiotic",PAL["red"])}
{param_slider("p-ab-onset","AB Onset Step",p.antibiotic_onset,0,200,1,"",PAL["red"])}
{param_slider("p-ab-max","Max Kill Rate",p.antibiotic_max,0,1,0.01,"",PAL["red"])}
{param_slider("p-ab-ramp","Ramp Steps",p.antibiotic_ramp,1,100,1,"Gradual mode only",PAL["red"])}
{param_slider("p-resA","Resistance A",p.resistance_A,0,1,0.01,"",PAL["red"])}
{param_slider("p-resB","Resistance B",p.resistance_B,0,1,0.01,"",PAL["yellow"])}
{param_slider("p-resC","Resistance C",p.resistance_C,0,1,0.01,"",PAL["green"])}

{sec_hdr("⏱","Simulation",PAL["orange"])}
{param_slider("p-tend","Total Time",int(p.t_end),50,1000,10,"",PAL["orange"])}

</div>"""

    # Survival status HTML
    def survival_badge(survived, name, color):
        s_color = PAL["green"] if survived else PAL["red"]
        s_txt   = "SURVIVED" if survived else "EXTINCT"
        return f"""
<div class="stat-box" style="border-color:{color}30">
  <div class="stat-box-lbl">{name}</div>
  <div class="stat-box-val" style="color:{color}">{summary[f"final_{name[-1]}"]}</div>
  <div class="stat-box-sub" style="color:{s_color}">{s_txt}</div>
</div>"""

    stats_html = f"""
<div class="stat-grid" style="margin-bottom:14px">
  {survival_badge(summary['survived'][0],"Gen A",PAL["red"])}
  {survival_badge(summary['survived'][1],"Gen B",PAL["yellow"])}
  {survival_badge(summary['survived'][2],"Gen C",PAL["green"])}
</div>
<div class="stat-grid">
  <div class="stat-box"><div class="stat-box-lbl">Peak Population</div><div class="stat-box-val">{summary["max_total"]}</div><div class="stat-box-sub" style="color:{PAL["muted"] if "muted" in PAL else MUTED}">at t={summary["peak_time"]}</div></div>
  <div class="stat-box"><div class="stat-box-lbl">Min Resource</div><div class="stat-box-val">{summary["min_resource"]}</div></div>
  <div class="stat-box"><div class="stat-box-lbl">Max Antibiotic</div><div class="stat-box-val" style="color:{PAL["red"]}">{summary["max_ab"]}</div></div>
  <div class="stat-box"><div class="stat-box-lbl">Quorum Active</div><div class="stat-box-val" style="color:{PAL["purple"]}">{summary["quorum_pct"]}</div><div class="stat-box-sub">of simulation</div></div>
  <div class="stat-box"><div class="stat-box-lbl">Biofilm Active</div><div class="stat-box-val" style="color:{PAL["orange"]}">{summary["biofilm_pct"]}</div><div class="stat-box-sub">of simulation</div></div>
  <div class="stat-box"><div class="stat-box-lbl">Coop/Comp</div><div class="stat-box-val" style="color:{PAL["purple"]}">{summary["coop_comp_ratio"]}</div><div class="stat-box-sub">ratio</div></div>
</div>"""

    def img_card(title, badge, color, b64_str, span=1):
        cls = "card full" if span > 1 else "card"
        return f"""
<div class="{cls}">
  <div class="card-hdr" style="border-top-color:{color}">
    <span class="card-title">{title}</span>
    <span class="badge" style="color:{color};background:{color}18;border:1px solid {color}30">{badge}</span>
  </div>
  <div class="card-body">
    <img src="data:image/png;base64,{b64_str}" alt="{title}">
  </div>
</div>"""

    def canvas_card(title, badge, color, cid, span=1):
        cls = "card full" if span > 1 else "card"
        return f"""
<div class="{cls}">
  <div class="card-hdr" style="border-top-color:{color}">
    <span class="card-title">{title}</span>
    <span class="badge" style="color:{color};background:{color}18;border:1px solid {color}30">{badge}</span>
  </div>
  <div class="card-body">
    <canvas id="{cid}" width="800" height="200" style="width:100%;height:auto"></canvas>
  </div>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>BactEvo — Bacterial Evolution Simulator</title>
<style>{CSS}</style>
</head>
<body>

<!-- TOP BAR -->
<div id="topbar">
  <div class="brand">
    <div class="brand-name">BACT<span>EVO</span></div>
    <div class="brand-sub">Sim Engine v5.0</div>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="kpi-lbl">Total Pop</div><div class="kpi-val" id="kpi-total">—</div></div>
    <div class="kpi"><div class="kpi-lbl">Resource</div><div class="kpi-val" id="kpi-res">—</div></div>
    <div class="kpi"><div class="kpi-lbl">Antibiotic</div><div class="kpi-val" id="kpi-ab" style="color:{PAL["red"]}">—</div></div>
    <div class="kpi"><div class="kpi-lbl">Quorum</div><div class="kpi-val" id="kpi-quorum" style="color:{PAL["purple"]}">—</div></div>
    <div class="kpi"><div class="kpi-lbl">Biofilm</div><div class="kpi-val" id="kpi-biofilm" style="color:{PAL["orange"]}">—</div></div>
    <div class="kpi"><div class="kpi-lbl">Extinct</div><div class="kpi-val" id="kpi-extinct">—</div></div>
    <div class="kpi"><div class="kpi-lbl">Environment</div><div class="kpi-val" style="color:{PAL["blue"]}">{summary["env_type"]}</div></div>
    <div class="kpi"><div class="kpi-lbl">Steps</div><div class="kpi-val">{summary["n_steps"]}</div></div>
  </div>
  <div style="color:{MUTED};font-size:8px;letter-spacing:.08em;margin-left:14px;border-left:1px solid {GRID_COL};padding-left:14px">
    Generated<br>{summary["gen_date"]}
  </div>
</div>

<!-- LAYOUT -->
<div id="layout">

  <!-- SIDEBAR -->
  <div id="sidebar">{sidebar_html}</div>

  <!-- CONTENT -->
  <div id="content">
    <!-- TABS -->
    <div class="tabs">
      <button class="tab-btn active" data-tab="population">Population</button>
      <button class="tab-btn" data-tab="fitness">Fitness</button>
      <button class="tab-btn" data-tab="mutation">Mutation</button>
      <button class="tab-btn" data-tab="resources">Resources</button>
      <button class="tab-btn" data-tab="social">Social</button>
      <button class="tab-btn" data-tab="spatial">Spatial</button>
      <button class="tab-btn" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="simulation">⚗ Simulation</button>
    </div>

    <!-- POPULATION -->
    <div id="panel-population" class="tab-panel">
      <div class="grid-2">
        {canvas_card("Genotype Densities","Live · Adjustable",PAL["red"],"cv-pop",span=2)}
        {canvas_card("Total Population","vs Carrying Capacity",PAL["green"],"cv-total")}
        {img_card("Proportion Over Time","100% Stacked",PAL["purple"],imgs["proportion"])}
        {img_card("Growth Rate (%)","Δ per Step",PAL["teal"],imgs["growth_rate"])}
        <div class="card">
          <div class="card-hdr" style="border-top-color:{PAL["teal"]}">
            <span class="card-title">Final State Summary</span>
            <span class="badge" style="color:{PAL["teal"]};background:{PAL["teal"]}18;border:1px solid {PAL["teal"]}30">Python ODE Results</span>
          </div>
          <div class="card-body">{stats_html}</div>
        </div>
      </div>
    </div>

    <!-- FITNESS -->
    <div id="panel-fitness" class="tab-panel" style="display:none">
      <div class="grid-2">
        {canvas_card("Fitness Trajectories","Live · Evolving",PAL["yellow"],"cv-fit",span=2)}
        {img_card("Fitness vs Population","Phase Space",PAL["purple"],imgs["fitness_vs_pop"])}
        {img_card("Fitness & Resistance","Final Comparison",PAL["orange"],imgs["fitness_bar"])}
        {img_card("Dominance Timeline","Categorical",PAL["teal"],imgs["dominance"],span=2)}
        {img_card("Phase Portrait  N_A vs N_B","Coloured by N_C",PAL["pink"],imgs["phase_portrait"],span=2)}
      </div>
    </div>

    <!-- MUTATION -->
    <div id="panel-mutation" class="tab-panel" style="display:none">
      <div class="grid-2">
        {canvas_card("Mutation Burden Over Time","Live · Accumulated",PAL["orange"],"cv-mut",span=2)}
        {img_card("Mutation Burden Rate","Δ per Step",PAL["yellow"],imgs["mut_rate"])}
        {img_card("Burden Distribution","Histogram + KDE",PAL["purple"],imgs["mut_dist"])}
      </div>
    </div>

    <!-- RESOURCES -->
    <div id="panel-resources" class="tab-panel" style="display:none">
      <div class="grid-2">
        {canvas_card("Resource & Antibiotic","Live · Dual Axis",PAL["blue"],"cv-res",span=2)}
        {img_card("Resource vs Population","Scatter · Colour=Time",PAL["teal"],imgs["res_vs_pop"])}
        {img_card("Antibiotic Kill Profiles","Per Genotype Resistance",PAL["red"],imgs["ab_profile"])}
      </div>
    </div>

    <!-- SOCIAL -->
    <div id="panel-social" class="tab-panel" style="display:none">
      <div class="grid-2">
        {canvas_card("Cooperation & Competition","Live · Indices",PAL["purple"],"cv-social",span=2)}
        {img_card("Social Dynamics + Ratio","Coop vs Comp",PAL["green"],imgs["social"],span=2)}
        {img_card("Quorum & Biofilm Events","ON/OFF Timeline",PAL["orange"],imgs["quorum_biofilm"],span=2)}
      </div>
    </div>

    <!-- SPATIAL -->
    <div id="panel-spatial" class="tab-panel" style="display:none">
      <div class="grid-2">
        {img_card("Spatial Distribution Snapshots",f"{p.spatial_size}×{p.spatial_size} Diffusion Grid",PAL["teal"],imgs["spatial"],span=2)}
      </div>
      <div style="margin-top:12px;padding:12px 14px;background:{SURFACE};border:1px solid {GRID_COL};border-radius:4px;font-size:9px;color:{MUTED};line-height:1.8">
        <span style="color:{PAL['teal']}">SPATIAL MODEL:</span>
        2-D diffusion grid ({p.spatial_size}×{p.spatial_size} cells) with periodic boundary conditions.
        Each genotype spreads via discrete Laplacian diffusion (D=0.08).
        Cell densities are rescaled each step to match the global ODE population totals.
        Shown at {len(imgs.get("spatial","")[:10])} evenly-spaced time snapshots.
      </div>
    </div>

    <!-- OVERVIEW -->
    <div id="panel-overview" class="tab-panel" style="display:none">
      <div class="grid-2">
        {img_card("All Variables — Normalised","10 Series · 0–1 Scale",PAL["teal"],imgs["all_norm"],span=2)}
        {img_card("Full Correlation Heatmap","12×12 Pearson Matrix",PAL["purple"],imgs["corr"],span=2)}
      </div>
    </div>


    <!-- SIMULATION -->
    <div id="panel-simulation" class="tab-panel" style="display:none">

      <!-- Sub-header -->
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;padding:10px 14px;
                  background:#0c1318;border:1px solid #1a2d38;border-radius:4px;
                  border-left:3px solid #00e5a0">
        <span style="font-size:9px;color:#00e5a0;letter-spacing:.16em;text-transform:uppercase;font-weight:700">
          ⚗ Interactive Simulation Engine
        </span>
        <span style="font-size:8px;color:#4a7a94;margin-left:8px">
          Adjust parameters below — charts update instantly
        </span>
        <button onclick="simRun()" style="margin-left:auto;padding:5px 14px;
          background:rgba(0,229,160,.12);border:1px solid rgba(0,229,160,.35);
          border-radius:3px;color:#00e5a0;font-family:'Space Mono',monospace;
          font-size:9px;letter-spacing:.12em;text-transform:uppercase;cursor:pointer">
          ▶ Re-Run
        </button>
      </div>

      <!-- KPI strip -->
      <div style="display:flex;gap:0;margin-bottom:16px;border:1px solid #1a2d38;border-radius:4px;overflow:hidden">
        <div class="skpi" id="skpi-total"  data-lbl="Total Pop"   style="color:#00e5a0"></div>
        <div class="skpi" id="skpi-res"    data-lbl="Resource"    style="color:#00b8ff"></div>
        <div class="skpi" id="skpi-ab"     data-lbl="Antibiotic"  style="color:#ff4d6d"></div>
        <div class="skpi" id="skpi-quorum" data-lbl="Quorum"      style="color:#c77dff"></div>
        <div class="skpi" id="skpi-biofilm"data-lbl="Biofilm"     style="color:#ff9f43"></div>
        <div class="skpi" id="skpi-domA"   data-lbl="Dom. Gen"    style="color:#ffd166"></div>
      </div>

      <div style="display:grid;grid-template-columns:300px 1fr;gap:14px">

        <!-- LEFT: parameter panel -->
        <div style="background:#0c1318;border:1px solid #1a2d38;border-radius:4px;
                    overflow-y:auto;max-height:calc(100vh - 230px);padding:14px 12px">

          <!-- Environment -->
          <div class="sp-sec" style="border-left-color:#00b8ff">🌍 ENVIRONMENT</div>
          <div class="sp-row"><span class="sp-lbl">Type</span></div>
          <select id="s-env" class="sp-sel" onchange="simRun()">
            <option value="rich">Resource-Rich</option>
            <option value="depleted">Resource-Depleted</option>
            <option value="antibiotic_gradual">Antibiotic — Gradual Ramp</option>
            <option value="antibiotic_spike">Antibiotic — Sudden Spike</option>
          </select>
          <div class="sp-slider" id="sg-ab" style="display:none">
            <div class="sp-row"><span class="sp-lbl">AB Onset Step</span><span class="sp-val" id="sv-ab-onset">30</span></div>
            <input type="range" id="s-ab-onset" min="0" max="200" step="1" value="30" oninput="sv('s-ab-onset','sv-ab-onset');simRun()">
            <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Max Kill Rate</span><span class="sp-val" id="sv-ab-max">0.30</span></div>
            <input type="range" id="s-ab-max" min="0" max="1" step="0.01" value="0.30" oninput="sv('s-ab-max','sv-ab-max');simRun()">
            <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Ramp Steps</span><span class="sp-val" id="sv-ab-ramp">20</span></div>
            <input type="range" id="s-ab-ramp" min="1" max="80" step="1" value="20" oninput="sv('s-ab-ramp','sv-ab-ramp');simRun()">
          </div>
          <div class="sp-row"><span class="sp-lbl">Resource Inflow</span><span class="sp-val" id="sv-inflow">2.0</span></div>
          <input type="range" id="s-inflow" min="0" max="10" step="0.1" value="2.0" oninput="sv('s-inflow','sv-inflow');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Initial Resource</span><span class="sp-val" id="sv-res0">100</span></div>
          <input type="range" id="s-res0" min="10" max="500" step="5" value="100" oninput="sv('s-res0','sv-res0');simRun()">

          <!-- Evolutionary Drivers -->
          <div class="sp-sec" style="border-left-color:#00e5a0;margin-top:14px">🧬 EVOLUTIONARY DRIVERS</div>
          <div class="sp-row"><span class="sp-lbl">Mutation Rate (μ)</span><span class="sp-val" id="sv-mu">0.001</span></div>
          <input type="range" id="s-mu" min="0" max="0.02" step="0.0001" value="0.001" oninput="sv('s-mu','sv-mu');simRun()">
          <div class="sp-desc">Per-capita mutation per step</div>
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">HGT Rate</span><span class="sp-val" id="sv-hgt">0.0005</span></div>
          <input type="range" id="s-hgt" min="0" max="0.005" step="0.0001" value="0.0005" oninput="sv('s-hgt','sv-hgt');simRun()">
          <div class="sp-desc">Horizontal gene transfer</div>
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Carrying Capacity (K)</span><span class="sp-val" id="sv-K">1000</span></div>
          <input type="range" id="s-K" min="100" max="5000" step="50" value="1000" oninput="sv('s-K','sv-K');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Generation Time</span><span class="sp-val" id="sv-gtime">1.0</span></div>
          <input type="range" id="s-gtime" min="0.1" max="5" step="0.1" value="1.0" oninput="sv('s-gtime','sv-gtime');simRun()">
          <div class="sp-desc">Scales growth rate (lower = faster)</div>
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Mutation Benefit</span><span class="sp-val" id="sv-mbenefit">0.02</span></div>
          <input type="range" id="s-mbenefit" min="0" max="0.2" step="0.005" value="0.02" oninput="sv('s-mbenefit','sv-mbenefit');simRun()">

          <!-- Fitness Landscape -->
          <div class="sp-sec" style="border-left-color:#ffd166;margin-top:14px">📈 FITNESS LANDSCAPE</div>
          <div class="sp-row"><span class="sp-lbl" style="color:#ff4d6d">Fitness A</span><span class="sp-val" id="sv-fA">1.00</span></div>
          <input type="range" id="s-fA" min="0.1" max="2" step="0.01" value="1.00" oninput="sv('s-fA','sv-fA');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl" style="color:#ffd166">Fitness B</span><span class="sp-val" id="sv-fB">0.95</span></div>
          <input type="range" id="s-fB" min="0.1" max="2" step="0.01" value="0.95" oninput="sv('s-fB','sv-fB');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl" style="color:#00e5a0">Fitness C</span><span class="sp-val" id="sv-fC">1.10</span></div>
          <input type="range" id="s-fC" min="0.1" max="2" step="0.01" value="1.10" oninput="sv('s-fC','sv-fC');simRun()">

          <!-- Initial Populations -->
          <div class="sp-sec" style="border-left-color:#4cc9f0;margin-top:14px">🔬 INITIAL POPULATIONS</div>
          <div class="sp-row"><span class="sp-lbl" style="color:#ff4d6d">Init A</span><span class="sp-val" id="sv-initA">200</span></div>
          <input type="range" id="s-initA" min="1" max="1000" step="10" value="200" oninput="sv('s-initA','sv-initA');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl" style="color:#ffd166">Init B</span><span class="sp-val" id="sv-initB">150</span></div>
          <input type="range" id="s-initB" min="1" max="1000" step="10" value="150" oninput="sv('s-initB','sv-initB');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl" style="color:#00e5a0">Init C</span><span class="sp-val" id="sv-initC">50</span></div>
          <input type="range" id="s-initC" min="1" max="1000" step="10" value="50" oninput="sv('s-initC','sv-initC');simRun()">

          <!-- Adversarial -->
          <div class="sp-sec" style="border-left-color:#ff4d6d;margin-top:14px">⚔️ ADVERSARIAL DYNAMICS</div>
          <div class="sp-row"><span class="sp-lbl">Bacteriocin Strength</span><span class="sp-val" id="sv-bact">0.003</span></div>
          <input type="range" id="s-bact" min="0" max="0.02" step="0.0001" value="0.003" oninput="sv('s-bact','sv-bact');simRun()">
          <div class="sp-desc">Dominant genotype suppresses rivals</div>
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Competition α</span><span class="sp-val" id="sv-alpha">0.80</span></div>
          <input type="range" id="s-alpha" min="0" max="2" step="0.05" value="0.80" oninput="sv('s-alpha','sv-alpha');simRun()">
          <div class="sp-desc">Inter-specific competition coefficient</div>
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Resource Consumption</span><span class="sp-val" id="sv-rcons">0.002</span></div>
          <input type="range" id="s-rcons" min="0" max="0.01" step="0.0001" value="0.002" oninput="sv('s-rcons','sv-rcons');simRun()">
          <div class="sp-desc">Per-capita resource use per step</div>

          <!-- Cooperative -->
          <div class="sp-sec" style="border-left-color:#c77dff;margin-top:14px">🤝 COOPERATIVE DYNAMICS</div>
          <div class="sp-row"><span class="sp-lbl">Public Good Benefit</span><span class="sp-val" id="sv-pg">0.05</span></div>
          <input type="range" id="s-pg" min="0" max="0.3" step="0.005" value="0.05" oninput="sv('s-pg','sv-pg');simRun()">
          <div class="sp-desc">Fitness bonus when quorum is reached</div>
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Quorum Threshold</span><span class="sp-val" id="sv-qt">300</span></div>
          <input type="range" id="s-qt" min="50" max="2000" step="25" value="300" oninput="sv('s-qt','sv-qt');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Biofilm Threshold</span><span class="sp-val" id="sv-bt">400</span></div>
          <input type="range" id="s-bt" min="100" max="3000" step="50" value="400" oninput="sv('s-bt','sv-bt');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl">Biofilm AB Protection</span><span class="sp-val" id="sv-bp">0.60</span></div>
          <input type="range" id="s-bp" min="0" max="1" step="0.01" value="0.60" oninput="sv('s-bp','sv-bp');simRun()">
          <div class="sp-desc">Reduces antibiotic efficacy in biofilm</div>

          <!-- Resistance per genotype -->
          <div class="sp-sec" style="border-left-color:#ff4d6d;margin-top:14px">💊 ANTIBIOTIC RESISTANCE</div>
          <div class="sp-row"><span class="sp-lbl" style="color:#ff4d6d">Resistance A</span><span class="sp-val" id="sv-resA">0.20</span></div>
          <input type="range" id="s-resA" min="0" max="1" step="0.01" value="0.20" oninput="sv('s-resA','sv-resA');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl" style="color:#ffd166">Resistance B</span><span class="sp-val" id="sv-resB">0.50</span></div>
          <input type="range" id="s-resB" min="0" max="1" step="0.01" value="0.50" oninput="sv('s-resB','sv-resB');simRun()">
          <div class="sp-row" style="margin-top:8px"><span class="sp-lbl" style="color:#00e5a0">Resistance C</span><span class="sp-val" id="sv-resC">0.80</span></div>
          <input type="range" id="s-resC" min="0" max="1" step="0.01" value="0.80" oninput="sv('s-resC','sv-resC');simRun()">

          <!-- Simulation control -->
          <div class="sp-sec" style="border-left-color:#ff9f43;margin-top:14px">⏱ SIMULATION CONTROL</div>
          <div class="sp-row"><span class="sp-lbl">Total Time Steps</span><span class="sp-val" id="sv-tend">200</span></div>
          <input type="range" id="s-tend" min="50" max="600" step="10" value="200" oninput="sv('s-tend','sv-tend');simRun()">

          <button onclick="simReset()" style="width:100%;margin-top:14px;padding:8px;
            background:transparent;border:1px solid #1a2d38;border-radius:3px;
            color:#4a7a94;font-family:'Space Mono',monospace;font-size:8px;
            letter-spacing:.12em;text-transform:uppercase;cursor:pointer">
            ↺ Reset to Defaults
          </button>
        </div>

        <!-- RIGHT: live charts -->
        <div style="display:flex;flex-direction:column;gap:12px;overflow-y:auto;max-height:calc(100vh - 230px)">

          <!-- Row 1: genotype densities (full width) -->
          <div class="scard" style="border-top-color:#ff4d6d">
            <div class="scard-hdr">
              <span class="scard-title">Genotype Densities</span>
              <span class="sbadge" style="color:#ff4d6d;border-color:#ff4d6d30;background:#ff4d6d10">Live · Euler</span>
            </div>
            <div class="scard-body"><canvas id="sc-pop" width="900" height="190" style="width:100%;height:auto"></canvas></div>
          </div>

          <!-- Row 2: resource + antibiotic | fitness -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div class="scard" style="border-top-color:#00b8ff">
              <div class="scard-hdr">
                <span class="scard-title">Resource & Antibiotic</span>
                <span class="sbadge" style="color:#00b8ff;border-color:#00b8ff30;background:#00b8ff10">Dual Axis</span>
              </div>
              <div class="scard-body"><canvas id="sc-res" width="440" height="170" style="width:100%;height:auto"></canvas></div>
            </div>
            <div class="scard" style="border-top-color:#ffd166">
              <div class="scard-hdr">
                <span class="scard-title">Fitness Landscape</span>
                <span class="sbadge" style="color:#ffd166;border-color:#ffd16630;background:#ffd16610">Evolving</span>
              </div>
              <div class="scard-body"><canvas id="sc-fit" width="440" height="170" style="width:100%;height:auto"></canvas></div>
            </div>
          </div>

          <!-- Row 3: mutation burden | coop+comp -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div class="scard" style="border-top-color:#ff9f43">
              <div class="scard-hdr">
                <span class="scard-title">Mutation Burden</span>
                <span class="sbadge" style="color:#ff9f43;border-color:#ff9f4330;background:#ff9f4310">Accumulated</span>
              </div>
              <div class="scard-body"><canvas id="sc-mut" width="440" height="170" style="width:100%;height:auto"></canvas></div>
            </div>
            <div class="scard" style="border-top-color:#c77dff">
              <div class="scard-hdr">
                <span class="scard-title">Cooperation vs Competition</span>
                <span class="sbadge" style="color:#c77dff;border-color:#c77dff30;background:#c77dff10">Social</span>
              </div>
              <div class="scard-body"><canvas id="sc-social" width="440" height="170" style="width:100%;height:auto"></canvas></div>
            </div>
          </div>

          <!-- Row 4: proportion stacked | total vs K -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div class="scard" style="border-top-color:#c77dff">
              <div class="scard-hdr">
                <span class="scard-title">Population Proportion</span>
                <span class="sbadge" style="color:#c77dff;border-color:#c77dff30;background:#c77dff10">100% Stacked</span>
              </div>
              <div class="scard-body"><canvas id="sc-prop" width="440" height="170" style="width:100%;height:auto"></canvas></div>
            </div>
            <div class="scard" style="border-top-color:#00e5a0">
              <div class="scard-hdr">
                <span class="scard-title">Total Pop vs Capacity</span>
                <span class="sbadge" style="color:#00e5a0;border-color:#00e5a030;background:#00e5a010">Logistic</span>
              </div>
              <div class="scard-body"><canvas id="sc-total" width="440" height="170" style="width:100%;height:auto"></canvas></div>
            </div>
          </div>

          <!-- Row 5: spatial heatmap (20x20 canvas grid) -->
          <div class="scard" style="border-top-color:#4cc9f0">
            <div class="scard-hdr">
              <span class="scard-title">Spatial Distribution (Final Step)</span>
              <span class="sbadge" style="color:#4cc9f0;border-color:#4cc9f030;background:#4cc9f010">20×20 Diffusion Grid</span>
            </div>
            <div class="scard-body" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;align-items:start">
              <div style="text-align:center">
                <div style="font-size:8px;color:#ff4d6d;letter-spacing:.1em;margin-bottom:6px">GENOTYPE A</div>
                <canvas id="sc-spA" width="160" height="160" style="width:100%;border:1px solid #1a2d38;border-radius:2px"></canvas>
              </div>
              <div style="text-align:center">
                <div style="font-size:8px;color:#ffd166;letter-spacing:.1em;margin-bottom:6px">GENOTYPE B</div>
                <canvas id="sc-spB" width="160" height="160" style="width:100%;border:1px solid #1a2d38;border-radius:2px"></canvas>
              </div>
              <div style="text-align:center">
                <div style="font-size:8px;color:#00e5a0;letter-spacing:.1em;margin-bottom:6px">GENOTYPE C</div>
                <canvas id="sc-spC" width="160" height="160" style="width:100%;border:1px solid #1a2d38;border-radius:2px"></canvas>
              </div>
            </div>
          </div>

        </div><!-- /right charts -->
      </div><!-- /grid -->
    </div><!-- /panel-simulation -->

  </div><!-- /content -->
</div><!-- /layout -->

<script>
{JS_ENGINE}
</script>
<script>
// ══════════════════════════════════════════════════════════════════════════
//  SIMULATION PANEL — full parameter set, 7 live charts + spatial grid
// ══════════════════════════════════════════════════════════════════════════

// Helper: update displayed slider value
function sv(sliderId, valId) {{
  document.getElementById(valId).textContent =
    parseFloat(document.getElementById(sliderId).value).toString();
}}

// Show/hide antibiotic sub-controls based on env type
document.getElementById('s-env').addEventListener('change', function() {{
  const show = this.value.startsWith('antibiotic');
  document.getElementById('sg-ab').style.display = show ? 'block' : 'none';
  simRun();
}});

// Read all sim-panel parameters
function simParams() {{
  const g = id => parseFloat(document.getElementById(id).value);
  const s = id => document.getElementById(id).value;
  return {{
    envType:            s('s-env'),
    inflow:             g('s-inflow'),
    initialResource:    g('s-res0'),
    resourceDecay:      0.05,
    mutationRate:       g('s-mu'),
    hgtRate:            g('s-hgt'),
    carryingCapacity:   g('s-K'),
    generationTime:     g('s-gtime'),
    mutationBenefit:    g('s-mbenefit'),
    initA:              g('s-initA'),
    initB:              g('s-initB'),
    initC:              g('s-initC'),
    fitnessA:           g('s-fA'),
    fitnessB:           g('s-fB'),
    fitnessC:           g('s-fC'),
    bacteriocin:        g('s-bact'),
    competitionAlpha:   g('s-alpha'),
    resourceConsumption:g('s-rcons'),
    publicGoodBenefit:  g('s-pg'),
    quorumThreshold:    g('s-qt'),
    biofilmThreshold:   g('s-bt'),
    biofilmProtection:  g('s-bp'),
    antibioticOnset:    g('s-ab-onset'),
    antibioticMax:      g('s-ab-max'),
    antibioticRamp:     g('s-ab-ramp'),
    resistanceA:        g('s-resA'),
    resistanceB:        g('s-resB'),
    resistanceC:        g('s-resC'),
    tEnd:               g('s-tend'),
    spatialSize:        20,
  }};
}}

// ── Euler simulation engine ──────────────────────────────────────────────
function simEngine(p) {{
  const dt = 0.5, steps = Math.round(p.tEnd / dt);
  const K = p.carryingCapacity, mu = p.mutationRate;
  let N   = [p.initA, p.initB, p.initC];
  let R   = p.initialResource;
  let mb  = [0, 0, 0];
  let fit = [p.fitnessA, p.fitnessB, p.fitnessC];
  const baseFit = [p.fitnessA, p.fitnessB, p.fitnessC];
  const res = [p.resistanceA, p.resistanceB, p.resistanceC];
  const hist = [];

  // Spatial grid 20x20x3
  const SZ = p.spatialSize;
  let grid = [];
  for (let r = 0; r < SZ; r++) {{
    grid.push([]);
    for (let c = 0; c < SZ; c++) {{
      const cx = SZ/2, cy = SZ/2;
      const dist = Math.sqrt((r-cy)**2+(c-cx)**2);
      const w = Math.exp(-dist*dist/(2*(SZ/4)**2));
      grid[r].push([p.initA*w*(0.8+Math.random()*0.4),
                    p.initB*w*0.7*(0.8+Math.random()*0.4),
                    p.initC*w*0.3*(0.8+Math.random()*0.4)]);
    }}
  }}

  for (let step = 0; step < steps; step++) {{
    const t = step * dt;

    // Antibiotic
    let ab = 0;
    if (p.envType === 'antibiotic_gradual' && t >= p.antibioticOnset)
      ab = Math.min(1, (t - p.antibioticOnset) / p.antibioticRamp);
    else if (p.envType === 'antibiotic_spike' && t >= p.antibioticOnset)
      ab = Math.max(0, 1 - (t - p.antibioticOnset) * 0.04);

    const envMult = p.envType === 'depleted' ? 0.3 : 1.0;
    const totN    = N.reduce((a,b)=>a+b,0);
    const quorum  = totN > p.quorumThreshold;
    const biofilm = totN > p.biofilmThreshold;
    const abEff   = ab * (biofilm ? (1-p.biofilmProtection) : 1);
    const resFac  = R > 0 ? R / (R + 20) : 0;
    const dom     = N.indexOf(Math.max(...N));

    const dN = [0,0,0];
    for (let i = 0; i < 3; i++) {{
      const Ni = Math.max(N[i], 0);
      const effFit = fit[i] + (quorum ? p.publicGoodBenefit : 0) + mb[i]*p.mutationBenefit;
      const compSum = N.reduce((s,Nj,j)=>s+(i===j?1:p.competitionAlpha)*Nj, 0);
      const growth  = (effFit/p.generationTime)*Ni*(1-compSum/K)*resFac;
      const mutFlux = N.reduce((s,Nj,j)=>j!==i?s+mu*(Nj-Ni):s, 0);
      const hgtFlux = N.reduce((s,Nj,j)=>j!==i&&Nj>Ni?s+p.hgtRate*Nj*Ni*0.01:s, 0);
      const bact    = i!==dom ? p.bacteriocin*N[dom]*Ni : 0;
      const kill    = abEff*(1-res[i])*p.antibioticMax*Ni;
      dN[i] = (growth+mutFlux+hgtFlux-bact-kill)*dt;
    }}

    const dR = (envMult*p.inflow - p.resourceConsumption*totN - p.resourceDecay*R)*dt;
    N   = N.map((v,i)=>Math.max(0.1, v+dN[i]));
    R   = Math.max(0, R+dR);
    mb  = mb.map((v,i)=>v+mu*N[i]*0.1*dt);
    fit = fit.map((v,i)=>v+0.02*(baseFit[i]-v)*dt);

    const tot  = N.reduce((a,b)=>a+b,0);
    const coop = quorum ? p.publicGoodBenefit : 0;
    const comp = p.bacteriocin * Math.max(...N);

    // Spatial diffusion (every 10 steps to keep it fast)
    if (step % 10 === 0) {{
      const newGrid = grid.map(row=>row.map(cell=>[...cell]));
      const D = 0.08;
      for (let r = 0; r < SZ; r++) {{
        for (let c = 0; c < SZ; c++) {{
          for (let g = 0; g < 3; g++) {{
            const rp=(r+1)%SZ,rm=(r-1+SZ)%SZ,cp=(c+1)%SZ,cm=(c-1+SZ)%SZ;
            const lap = grid[rp][c][g]+grid[rm][c][g]+grid[r][cp][g]+grid[r][cm][g]-4*grid[r][c][g];
            newGrid[r][c][g] = Math.max(0, grid[r][c][g]+D*lap*dt);
          }}
        }}
      }}
      // Rescale to match global pops
      for (let g = 0; g < 3; g++) {{
        let cellSum = 0;
        for (let r=0;r<SZ;r++) for (let c=0;c<SZ;c++) cellSum+=newGrid[r][c][g];
        if (cellSum > 0) {{
          const scale = N[g]/cellSum;
          for (let r=0;r<SZ;r++) for (let c=0;c<SZ;c++) newGrid[r][c][g]*=scale;
        }}
      }}
      grid = newGrid;
    }}

    hist.push({{t:Math.round(t*10)/10,
               pA:Math.round(N[0]),pB:Math.round(N[1]),pC:Math.round(N[2]),
               tot:Math.round(tot), res:Math.round(R*10)/10,
               ab:Math.round(abEff*1000)/1000,
               fA:Math.round(fit[0]*1000)/1000,
               fB:Math.round(fit[1]*1000)/1000,
               fC:Math.round(fit[2]*1000)/1000,
               mA:Math.round(mb[0]*100)/100,
               mB:Math.round(mb[1]*100)/100,
               mC:Math.round(mb[2]*100)/100,
               quorum:quorum?1:0, biofilm:biofilm?1:0,
               coop, comp,
               propA:tot>0?N[0]/tot:0, propB:tot>0?N[1]/tot:0, propC:tot>0?N[2]/tot:0}});
  }}
  return {{ hist, grid, p }};
}}

// ── Canvas drawing helpers ───────────────────────────────────────────────
const _BG='#060a0f',_SRF='#0c1318',_GRD='#1a2d38',_MUT='#4a7a94',_TXT='#d4eaf7';
const _GC=['#ff4d6d','#ffd166','#00e5a0'];

function _clearCanvas(canvas) {{
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle=_SRF; ctx.fillRect(0,0,canvas.width,canvas.height);
  return ctx;
}}

function _drawLines(canvas, datasets, opts={{}}) {{
  const ctx=_clearCanvas(canvas);
  const W=canvas.width, H=canvas.height;
  const P={{l:52,r:14,t:10,b:30}};
  const cw=W-P.l-P.r, ch=H-P.t-P.b;

  let yMin=opts.yMin!==undefined?opts.yMin:Infinity;
  let yMax=opts.yMax!==undefined?opts.yMax:-Infinity;
  let xMax=-Infinity;
  datasets.forEach(ds=>ds.data.forEach(d=>{{
    if(opts.yMin===undefined&&d.y<yMin)yMin=d.y;
    if(opts.yMax===undefined&&d.y>yMax)yMax=d.y;
    if(d.x>xMax)xMax=d.x;
  }}));
  if(yMin===yMax){{yMin-=1;yMax+=1;}}
  const tx=x=>P.l+(x/(xMax||1))*cw;
  const ty=y=>P.t+ch-(y-yMin)/(yMax-yMin)*ch;

  // Grid
  ctx.strokeStyle=_GRD; ctx.lineWidth=0.4;
  for(let i=0;i<=4;i++){{
    const yv=yMin+i*(yMax-yMin)/4, y=ty(yv);
    ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(P.l+cw,y);ctx.stroke();
    ctx.fillStyle=_MUT;ctx.font="8px 'Space Mono',monospace";ctx.textAlign='right';
    const lbl=yv>=1e6?`${{(yv/1e6).toFixed(1)}}M`:yv>=1e3?`${{(yv/1e3).toFixed(0)}}K`:yv.toFixed(yv<10?2:0);
    ctx.fillText(lbl,P.l-3,y+3);
  }}
  for(let i=0;i<=5;i++){{
    const xv=i*xMax/5, x=tx(xv);
    ctx.beginPath();ctx.moveTo(x,P.t);ctx.lineTo(x,P.t+ch);ctx.stroke();
    ctx.fillStyle=_MUT;ctx.font="8px 'Space Mono',monospace";ctx.textAlign='center';
    ctx.fillText(xv.toFixed(0),x,P.t+ch+14);
  }}
  if(opts.xlabel){{ctx.fillStyle=_MUT;ctx.font="8px 'Space Mono',monospace";ctx.textAlign='center';ctx.fillText(opts.xlabel,P.l+cw/2,H-2);}}

  // Optional ref line (K)
  if(opts.refY!==undefined){{
    const ry=ty(opts.refY);
    ctx.strokeStyle=_MUT;ctx.lineWidth=1;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(P.l,ry);ctx.lineTo(P.l+cw,ry);ctx.stroke();
    ctx.setLineDash([]);
  }}

  datasets.forEach(ds=>{{
    const pts=ds.data; if(!pts.length)return;
    // fill
    ctx.beginPath();ctx.moveTo(tx(pts[0].x),ty(0));
    pts.forEach(p=>ctx.lineTo(tx(p.x),ty(p.y)));
    ctx.lineTo(tx(pts[pts.length-1].x),ty(0));ctx.closePath();
    ctx.fillStyle=ds.color+'18';ctx.fill();
    // line
    ctx.beginPath();pts.forEach((p,i)=>i===0?ctx.moveTo(tx(p.x),ty(p.y)):ctx.lineTo(tx(p.x),ty(p.y)));
    ctx.strokeStyle=ds.color;ctx.lineWidth=1.8;ctx.stroke();
  }});

  // Legend
  let lx=P.l+6;
  datasets.forEach(ds=>{{
    ctx.fillStyle=ds.color;ctx.fillRect(lx,P.t+2,12,3);
    ctx.fillStyle=_TXT;ctx.font="7px 'Space Mono',monospace";ctx.textAlign='left';
    ctx.fillText(ds.label,lx+15,P.t+8);
    lx+=ctx.measureText(ds.label).width+28;
  }});
}}

function _drawStacked(canvas, hist) {{
  const ctx=_clearCanvas(canvas);
  const W=canvas.width, H=canvas.height;
  const P={{l:40,r:10,t:10,b:28}};
  const cw=W-P.l-P.r, ch=H-P.t-P.b;
  const n=hist.length;
  const tx=i=>P.l+(i/(n-1))*cw;

  // Grid
  ctx.strokeStyle=_GRD;ctx.lineWidth=0.4;
  [0,0.25,0.5,0.75,1].forEach(v=>{{
    const y=P.t+ch-v*ch;
    ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(P.l+cw,y);ctx.stroke();
    ctx.fillStyle=_MUT;ctx.font="8px 'Space Mono',monospace";ctx.textAlign='right';
    ctx.fillText((v*100).toFixed(0)+'%',P.l-3,y+3);
  }});

  // Stacked areas
  const layers=[
    hist.map(d=>d.propA),
    hist.map(d=>d.propA+d.propB),
    hist.map(d=>1),
  ];
  const colors=[_GC[0],_GC[1],_GC[2]];
  for(let g=2;g>=0;g--){{
    ctx.beginPath();
    for(let i=0;i<n;i++){{
      const y=P.t+ch-(layers[g][i])*ch;
      i===0?ctx.moveTo(tx(i),y):ctx.lineTo(tx(i),y);
    }}
    const base=g===0?null:layers[g-1];
    if(base){{for(let i=n-1;i>=0;i--) ctx.lineTo(tx(i),P.t+ch-base[i]*ch);}}
    else{{ctx.lineTo(tx(n-1),P.t+ch);ctx.lineTo(tx(0),P.t+ch);}}
    ctx.closePath();ctx.fillStyle=colors[g]+'b0';ctx.fill();
  }}
}}

function _drawSpatial(canvas, grid, genIdx) {{
  const ctx=_clearCanvas(canvas);
  const SZ=grid.length, CW=canvas.width/SZ, CH=canvas.height/SZ;
  const rgbs=[[255,77,109],[255,209,102],[0,229,160]];
  let mx=0;
  for(let r=0;r<SZ;r++) for(let c=0;c<SZ;c++) if(grid[r][c][genIdx]>mx) mx=grid[r][c][genIdx];
  for(let r=0;r<SZ;r++){{
    for(let c=0;c<SZ;c++){{
      const v=mx>0?grid[r][c][genIdx]/mx:0;
      const dark=rgbs[genIdx], bg=[6,10,15];
      const rgb=dark.map((d,i)=>Math.round(bg[i]+v*(d-bg[i])));
      ctx.fillStyle=`rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},${{0.2+v*0.8}})`;
      ctx.fillRect(c*CW, r*CH, CW, CH);
    }}
  }}
}}

// ── Update KPI strip ────────────────────────────────────────────────────
function _updateSkpis(last, p) {{
  function setKpi(id, lbl, val, col) {{
    const el=document.getElementById(id);
    if(!el)return;
    el.innerHTML=`<div style="font-size:8px;color:#4a7a94;letter-spacing:.08em;text-transform:uppercase">${{lbl}}</div>
                  <div style="font-size:13px;font-weight:700;color:${{col}};margin-top:2px">${{val}}</div>`;
  }}
  const dom=['A','B','C'][[last.pA,last.pB,last.pC].indexOf(Math.max(last.pA,last.pB,last.pC))];
  setKpi('skpi-total', 'Total Pop',   last.tot.toLocaleString(),            '#00e5a0');
  setKpi('skpi-res',   'Resource',    last.res+' mM',                       '#00b8ff');
  setKpi('skpi-ab',    'Antibiotic',  (last.ab*100).toFixed(0)+'%',         '#ff4d6d');
  setKpi('skpi-quorum','Quorum',      last.quorum?'ACTIVE':'—',             '#c77dff');
  setKpi('skpi-biofilm','Biofilm',    last.biofilm?'ACTIVE':'—',            '#ff9f43');
  setKpi('skpi-domA',  'Dom. Gen',    'Gen '+dom,                           '#ffd166');
}}

// ── Main render ─────────────────────────────────────────────────────────
function simRender({{hist, grid, p}}) {{
  const mk = (key) => hist.map(d=>({{x:d.t, y:d[key]}}));

  // Population densities
  _drawLines(document.getElementById('sc-pop'), [
    {{label:'Genotype A', color:_GC[0], data:mk('pA')}},
    {{label:'Genotype B', color:_GC[1], data:mk('pB')}},
    {{label:'Genotype C', color:_GC[2], data:mk('pC')}},
  ], {{xlabel:'Time', refY:p.carryingCapacity}});

  // Resource + antibiotic (scaled to same axis)
  const abScale=Math.max(...hist.map(d=>d.res))||1;
  _drawLines(document.getElementById('sc-res'), [
    {{label:'Resource (mM)', color:'#00b8ff', data:mk('res')}},
    {{label:'Antibiotic ×'+abScale.toFixed(0), color:'#ff4d6d',
     data:hist.map(d=>({{x:d.t, y:d.ab*abScale}}))}},
  ], {{xlabel:'Time'}});

  // Fitness
  _drawLines(document.getElementById('sc-fit'), [
    {{label:'Fitness A', color:_GC[0], data:mk('fA')}},
    {{label:'Fitness B', color:_GC[1], data:mk('fB')}},
    {{label:'Fitness C', color:_GC[2], data:mk('fC')}},
  ], {{xlabel:'Time'}});

  // Mutation burden
  _drawLines(document.getElementById('sc-mut'), [
    {{label:'Burden A', color:_GC[0], data:mk('mA')}},
    {{label:'Burden B', color:_GC[1], data:mk('mB')}},
    {{label:'Burden C', color:_GC[2], data:mk('mC')}},
  ], {{xlabel:'Time'}});

  // Social
  _drawLines(document.getElementById('sc-social'), [
    {{label:'Cooperation', color:'#00e5a0', data:mk('coop')}},
    {{label:'Competition', color:'#ff4d6d', data:mk('comp')}},
  ], {{xlabel:'Time'}});

  // Proportion stacked
  _drawStacked(document.getElementById('sc-prop'), hist);

  // Total vs K
  _drawLines(document.getElementById('sc-total'), [
    {{label:'Total Pop', color:'#00e5a0', data:mk('tot')}},
  ], {{xlabel:'Time', refY:p.carryingCapacity}});

  // Spatial heatmaps
  _drawSpatial(document.getElementById('sc-spA'), grid, 0);
  _drawSpatial(document.getElementById('sc-spB'), grid, 1);
  _drawSpatial(document.getElementById('sc-spC'), grid, 2);

  // KPI strip
  _updateSkpis(hist[hist.length-1], p);
}}

// ── Debounced run ────────────────────────────────────────────────────────
let _simTimer;
function simRun() {{
  clearTimeout(_simTimer);
  _simTimer = setTimeout(()=>{{
    const p = simParams();
    const result = simEngine(p);
    simRender(result);
  }}, 100);
}}

// ── Reset to defaults ────────────────────────────────────────────────────
function simReset() {{
  const defaults = {{
    's-env':'rich','s-inflow':2.0,'s-res0':100,'s-mu':0.001,'s-hgt':0.0005,
    's-K':1000,'s-gtime':1.0,'s-mbenefit':0.02,
    's-initA':200,'s-initB':150,'s-initC':50,
    's-fA':1.00,'s-fB':0.95,'s-fC':1.10,
    's-bact':0.003,'s-alpha':0.80,'s-rcons':0.002,
    's-pg':0.05,'s-qt':300,'s-bt':400,'s-bp':0.60,
    's-ab-onset':30,'s-ab-max':0.30,'s-ab-ramp':20,
    's-resA':0.20,'s-resB':0.50,'s-resC':0.80,'s-tend':200,
  }};
  Object.entries(defaults).forEach(([id,val])=>{{
    const el=document.getElementById(id);
    if(el){{el.value=val;}}
  }});
  // Refresh displayed values
  [['s-inflow','sv-inflow'],['s-res0','sv-res0'],['s-mu','sv-mu'],
   ['s-hgt','sv-hgt'],['s-K','sv-K'],['s-gtime','sv-gtime'],['s-mbenefit','sv-mbenefit'],
   ['s-initA','sv-initA'],['s-initB','sv-initB'],['s-initC','sv-initC'],
   ['s-fA','sv-fA'],['s-fB','sv-fB'],['s-fC','sv-fC'],
   ['s-bact','sv-bact'],['s-alpha','sv-alpha'],['s-rcons','sv-rcons'],
   ['s-pg','sv-pg'],['s-qt','sv-qt'],['s-bt','sv-bt'],['s-bp','sv-bp'],
   ['s-ab-onset','sv-ab-onset'],['s-ab-max','sv-ab-max'],['s-ab-ramp','sv-ab-ramp'],
   ['s-resA','sv-resA'],['s-resB','sv-resB'],['s-resC','sv-resC'],['s-tend','sv-tend']
  ].forEach(([s,v])=>sv(s,v));
  document.getElementById('sg-ab').style.display='none';
  simRun();
}}

// Auto-run when tab is first clicked
let _simInitDone = false;
document.querySelectorAll('.tab-btn').forEach(btn=>{{
  if(btn.dataset.tab==='simulation'){{
    btn.addEventListener('click',()=>{{
      if(!_simInitDone){{ _simInitDone=true; simRun(); }}
    }});
  }}
}});
</script>

<style>
.skpi{{flex:1;padding:8px 12px;border-right:1px solid #1a2d38;background:#0c1318;text-align:center}}
.skpi:last-child{{border-right:none}}
.sp-sec{{font-size:8px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;
        padding:6px 0 6px 8px;border-left:3px solid;margin-bottom:8px;color:#d4eaf7}}
.sp-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px}}
.sp-lbl{{font-size:8px;color:#6a9ab4;text-transform:uppercase;letter-spacing:.06em}}
.sp-val{{font-size:10px;color:#d4eaf7;font-weight:700;font-family:'Space Mono',monospace}}
.sp-desc{{font-size:8px;color:#4a7a94;margin-top:1px;margin-bottom:8px;line-height:1.4}}
.sp-sel{{width:100%;background:#162430;border:1px solid #1a2d38;border-radius:3px;
        color:#d4eaf7;font-family:'Space Mono',monospace;font-size:9px;
        padding:5px 7px;outline:none;cursor:pointer;margin-bottom:10px}}
.sp-slider{{background:#0c1820;border:1px solid #1a2d38;border-radius:3px;
           padding:10px;margin-bottom:10px}}
.scard{{background:#0c1318;border:1px solid #1a2d38;border-radius:4px;
       overflow:hidden;border-top:2px solid}}
.scard-hdr{{padding:8px 12px 6px;border-bottom:1px solid #1a2d38;background:#111b22;
           display:flex;justify-content:space-between;align-items:center}}
.scard-title{{font-family:'Syne',sans-serif;font-size:11px;font-weight:700;color:#fff}}
.sbadge{{font-size:7px;letter-spacing:.1em;text-transform:uppercase;
        padding:2px 6px;border-radius:2px;border:1px solid}}
.scard-body{{padding:10px 8px;background:#0c1318}}
</style>

</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="BactEvo — Bacterial Evolution Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", choices=["rich","depleted","antibiotic_gradual","antibiotic_spike"],
                        default="rich", dest="env_type")
    parser.add_argument("--mu",  type=float, default=0.001,  dest="mutation_rate")
    parser.add_argument("--K",   type=float, default=1000.0, dest="carrying_capacity")
    parser.add_argument("--t",   type=float, default=200.0,  dest="t_end")
    parser.add_argument("--initA", type=float, default=200.0)
    parser.add_argument("--initB", type=float, default=150.0)
    parser.add_argument("--initC", type=float, default=50.0)
    parser.add_argument("--fA",  type=float, default=1.00,   dest="fitness_A")
    parser.add_argument("--fB",  type=float, default=0.95,   dest="fitness_B")
    parser.add_argument("--fC",  type=float, default=1.10,   dest="fitness_C")
    parser.add_argument("--ab-onset", type=float, default=30.0,  dest="antibiotic_onset")
    parser.add_argument("--ab-max",   type=float, default=0.30,  dest="antibiotic_max")
    parser.add_argument("--out", type=str, default="report.html", dest="output")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    p = SimParams(
        env_type          = args.env_type,
        mutation_rate     = args.mutation_rate,
        carrying_capacity = args.carrying_capacity,
        t_end             = args.t_end,
        init_A            = args.initA,
        init_B            = args.initB,
        init_C            = args.initC,
        fitness_A         = args.fitness_A,
        fitness_B         = args.fitness_B,
        fitness_C         = args.fitness_C,
        antibiotic_onset  = args.antibiotic_onset,
        antibiotic_max    = args.antibiotic_max,
    )

    print("=" * 60)
    print("  BactEvo — Bacterial Evolution Simulation")
    print("=" * 60)
    print(f"  Environment : {p.env_type}")
    print(f"  Genotypes   : A={p.init_A}, B={p.init_B}, C={p.init_C}")
    print(f"  Fitness     : A={p.fitness_A}, B={p.fitness_B}, C={p.fitness_C}")
    print(f"  Capacity    : K={p.carrying_capacity}")
    print(f"  Mutation    : μ={p.mutation_rate}, HGT={p.hgt_rate}")
    print(f"  Duration    : t=[0, {p.t_end}]")
    print()

    # ── CSV LOADER ────────────────────────────────────────────────────────────
    import pandas as pd

    df = pd.read_csv("simulation_metrics.csv")

    if len(df.dropna(how="all")) > 0:
        # CSV has data — build res directly from it
        print(f"  [CSV] Loaded {len(df)} rows from 'simulation_metrics.csv'")
        t_arr = df["time_step"].to_numpy(dtype=float)
        total = df["total_population"].to_numpy(dtype=float)
        res = dict(
            t        = t_arr,
            N        = np.vstack([
                           df["genotype_A_density"].to_numpy(dtype=float),
                           df["genotype_B_density"].to_numpy(dtype=float),
                           df["genotype_C_density"].to_numpy(dtype=float),
                       ]),
            R        = df["resource_concentration"].to_numpy(dtype=float),
            mb       = np.vstack([
                           df["mutation_frequency"].to_numpy(dtype=float) * df["genotype_A_density"].to_numpy(dtype=float),
                           df["mutation_frequency"].to_numpy(dtype=float) * df["genotype_B_density"].to_numpy(dtype=float),
                           df["mutation_frequency"].to_numpy(dtype=float) * df["genotype_C_density"].to_numpy(dtype=float),
                       ]),
            fit      = np.vstack([
                           np.gradient(df["genotype_A_density"].to_numpy(dtype=float), t_arr).clip(0) + p.fitness_A,
                           np.gradient(df["genotype_B_density"].to_numpy(dtype=float), t_arr).clip(0) + p.fitness_B,
                           np.gradient(df["genotype_C_density"].to_numpy(dtype=float), t_arr).clip(0) + p.fitness_C,
                       ]),
            total_N  = total,
            ab       = np.zeros(len(t_arr)),
            quorum   = (total > p.quorum_threshold).astype(float),
            biofilm  = (total > p.biofilm_threshold).astype(float),
            prop     = np.nan_to_num(np.vstack([
                           df["genotype_A_density"].to_numpy(float) / np.where(total > 0, total, 1),
                           df["genotype_B_density"].to_numpy(float) / np.where(total > 0, total, 1),
                           df["genotype_C_density"].to_numpy(float) / np.where(total > 0, total, 1),
                       ])),
            coop_idx = df["cooperation_index"].to_numpy(dtype=float),
            comp_idx = df["competition_index"].to_numpy(dtype=float),
            params   = p,
        )
        print(f"        ✓  {len(t_arr)} time points from CSV")
    else:
        # CSV is empty — run ODE and fill the CSV
        print("  [CSV] 'simulation_metrics.csv' is empty — running ODE simulation")
        print("  [1/4] Running ODE simulation (scipy RK45)...")
        res = run_simulation(p)
        print(f"        ✓  {len(res['t'])} time points")

        df_out = pd.DataFrame({
            "time_step":              res["t"],
            "total_population":       res["total_N"],
            "resource_concentration": res["R"],
            "genotype_A_density":     res["N"][0],
            "genotype_B_density":     res["N"][1],
            "genotype_C_density":     res["N"][2],
            "mutation_frequency":     res["mb"][0] / np.where(res["N"][0] > 0, res["N"][0], 1),
            "cooperation_index":      res["coop_idx"],
            "competition_index":      res["comp_idx"],
        })
        df_out.to_csv("simulation_metrics.csv", index=False, float_format="%.6f")
        print("        ✓  Exported simulation data → simulation_metrics.csv")

    print("  [2/4] Running spatial diffusion simulation...")
    snapshots = run_spatial(p, res, n_snaps=8)
    print(f"        ✓  {len(snapshots)} spatial snapshots ({p.spatial_size}×{p.spatial_size} grid)")

    print("  [3/4] Rendering charts...")
    imgs = {
        "densities":      chart_genotype_densities(res),
        "total_pop":      chart_total_population(res),
        "proportion":     chart_proportion(res),
        "growth_rate":    chart_growth_rate(res),
        "fitness":        chart_fitness(res),
        "fitness_vs_pop": chart_fitness_vs_population(res),
        "fitness_bar":    chart_fitness_bar(res),
        "mut_burden":     chart_mutation_burden(res),
        "mut_rate":       chart_mutation_rate_over_time(res),
        "mut_dist":       chart_mutation_distribution(res),
        "resource":       chart_resource(res),
        "res_vs_pop":     chart_resource_vs_population(res),
        "ab_profile":     chart_antibiotic_profile(res),
        "social":         chart_social(res),
        "quorum_biofilm": chart_quorum_biofilm(res),
        "corr":           chart_correlation_heatmap(res),
        "spatial":        chart_spatial_panels(snapshots),
        "dominance":      chart_dominance_timeline(res),
        "phase_portrait": chart_phase_portrait(res),
        "all_norm":       chart_all_normalised(res),
    }
    print(f"        ✓  {len(imgs)} charts rendered")

    print("  [4/4] Building HTML report...")
    summary = compute_summary(res, p)
    html = build_html(imgs, summary, p)

    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(out) / 1e6
    print(f"        ✓  {out}  ({size_mb:.1f} MB)")
    print()
    print("  RESULTS")
    print(f"  Peak population : {summary['max_total']}  at t={summary['peak_time']}")
    print(f"  Gen A final     : {summary['final_A']}  {'✓' if summary['survived'][0] else '✗ EXTINCT'}")
    print(f"  Gen B final     : {summary['final_B']}  {'✓' if summary['survived'][1] else '✗ EXTINCT'}")
    print(f"  Gen C final     : {summary['final_C']}  {'✓' if summary['survived'][2] else '✗ EXTINCT'}")
    print(f"  Coop/Comp ratio : {summary['coop_comp_ratio']}")
    print()
    print(f"  Open {out} in any browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
