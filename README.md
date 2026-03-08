# 🦠 BactEvo — Bacterial Evolution Simulation Framework

> 🧬 Built for **HackBIO '26** | Kamand Bioengineering Group, IIT Mandi
> 🔬 Track: Computational Systems Biology

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![HackBIO](https://img.shields.io/badge/HackBIO-2026-purple?style=for-the-badge)

---

## 📹 Demo Video

[![BactEvo Demo](https://img.shields.io/badge/▶%20Watch%20Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](YOUR_VIDEO_LINK_HERE)

> Replace YOUR_VIDEO_LINK_HERE with your actual video URL

---

## 🌟 Overview

BactEvo is an **agent-based + ODE hybrid** simulation framework that models the survival, evolution, and behavioral dynamics of a bacterial population in resource-constrained environments. It tracks **genotype competition**, **mutation spread**, **quorum sensing**, **horizontal gene transfer (HGT)**, and **antibiotic stress responses** — outputting a fully self-contained interactive HTML report and structured CSV for analysis.

---

## 📁 File Structure

    BactEvo/
    ├── 🧫 agent.py                  # Core simulation engine (ODE + ABM hybrid)
    ├── ⚙️  simulate.py               # Lightweight CSV generator
    ├── 📊 visualize.py              # Chart generator — reads CSV, outputs HTML report
    ├── 📄 simulation_metrics.csv    # Time-series output (1000 time steps, 9 metrics)
    ├── 🌐 report.html               # Full interactive HTML report with embedded charts
    ├── 📈 chart.html                # Standalone chart output
    └── 📖 README.md

---

## 🚀 Quickstart

**1️⃣ Install dependencies**

    pip install numpy scipy matplotlib seaborn pandas

**2️⃣ Run the full simulation**

    python agent.py                          # default (resource-rich)
    python agent.py --env antibiotic_spike   # sudden antibiotic dose
    python agent.py --env antibiotic_gradual
    python agent.py --env depleted
    python agent.py --help

> Output: report.html — open in any browser, no server needed.

**3️⃣ Run the lightweight CSV generator**

    python simulate.py

> Output: simulation_metrics.csv with 1000 rows.

**4️⃣ Generate charts from CSV**

    python visualize.py

> Output: report.html with all charts embedded as base64 PNG.

---

## 📊 Simulation Metrics (CSV Schema)

| Column | Description |
|---|---|
| ⏱️ time_step | Current simulation epoch |
| 🦠 total_population | Absolute count of living bacterial cells |
| 🧪 resource_concentration | Available nutrient level in the environment (mM) |
| 🔴 genotype_A_density | Relative frequency of Genotype A |
| 🟡 genotype_B_density | Relative frequency of Genotype B |
| 🟢 genotype_C_density | Relative frequency of Genotype C |
| 🧬 mutation_frequency | Rate of novel trait appearance per time step |
| 🤝 cooperation_index | Quantitative measure of friendly behavior (biofilm, public goods) |
| ⚔️ competition_index | Quantitative measure of adversarial behavior (toxin/bacteriocin levels) |

---

## 🧪 Biological Model

### 🌍 Environment Types

| Type | Description |
|---|---|
| 🟢 Rich | High resource inflow, no stressors |
| 🟠 Depleted | Low resource inflow, competition intensifies |
| 🔴 Antibiotic Gradual | Slow-rising antibiotic; tests adaptive evolution |
| 💀 Antibiotic Spike | Sudden lethal dose; tests resistance and recovery |

### 🔬 Evolutionary Mechanisms

- 🧫 **Monod Growth Kinetics** — Resource-dependent growth rates
- 🏔️ **Fitness Landscapes** — Genotype-specific fitness scores updated each step
- 🧬 **Mutation** — Random genotype transitions at configurable mutation rates
- 🔗 **Horizontal Gene Transfer (HGT)** — Gene sharing between neighbouring cells
- 📡 **Quorum Sensing** — Cooperative behaviour triggered above population density thresholds
- ☠️ **Bacteriocin Production** — Adversarial toxin release modelled as competition index

### 📈 Population Phases Tracked

    Lag → Log (Exponential) → Stationary → Death

All visible in population vs. time curves.

---

## ⚙️ Key Parameters

| Parameter | Default | Description |
|---|---|---|
| env_type | rich | Environment preset |
| mutation_rate | 0.001 | Probability of mutation per cell per step |
| hgt_rate | 0.0005 | Horizontal gene transfer rate |
| carrying_capacity | 1000.0 | Maximum sustainable population |
| initial_resource | 100.0 | Starting nutrient concentration (mM) |
| inflow | 2.0 | Resource replenishment rate (mM/t) |
| init_A / B / C | 200/150/50 | Initial population of each genotype |

---

## 📉 Visualizations Produced

- 📊 **Population Dynamics** — Total population and per-genotype curves over time
- 🧬 **Genotype Evolution** — Stacked area chart showing genotype frequency shifts
- 💧 **Resource Depletion** — Nutrient concentration matched against bacterial growth phases
- 🔀 **Mutation Frequency** — Novel trait appearance rate over time
- 🤝 **Cooperation vs. Competition** — Dual-axis index plots showing social dynamics
- 🌀 **Phase Diagrams** — Population vs. resource phase-space trajectories
- 🗺️ **Spatial Maps** — Colony formation and adversarial boundary visualization

> All charts embedded in report.html — fully offline, no dependencies needed to view.

---

## 🧠 Biological Assumptions

1. Bacterial growth follows Monod kinetics: **μ = μ_max × (S / (K_s + S))**
2. Three competing genotypes with distinct fitness coefficients
3. Resource is finite and consumed proportionally to population size
4. Antibiotic stress introduces a fitness penalty applied differentially by genotype
5. Cooperation (biofilm/public goods) modelled as a density-dependent threshold effect
6. HGT allows resistance genes to transfer between genotypes at a low background rate
7. Carrying capacity enforces logistic growth ceiling

---

## ⚠️ Limitations and Future Work

- Currently models 3 discrete genotypes; future versions could implement continuous fitness landscapes
- Spatial modelling is 2D grid-based; 3D biofilm modelling is a planned extension
- Antibiotic pharmacokinetics (PK/PD) are simplified; full MIC curve modelling is future work
- Reinforcement Learning agents for adaptive bacterial behaviour are a planned enhancement

---

## 👥 Team

See TEAM.txt for team member details.

---

## 🏆 Hackathon

**HackBIO '26** | Organized by Kamand Bioengineering Group, IIT Mandi
Problem Domain: Computational Systems Biology
Contact: Devansh Garg — +91 99103 29901
