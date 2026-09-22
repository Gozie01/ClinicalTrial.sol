"""Figure 6 - E6 Warm/Cold Federated Fairness De-confound (Cimas, K=23).

Scientific question: Is the FedAvg fairness benefit driven by ongoing gradient
sharing or by a single shared initialisation?

Four conditions x 5 seeds x 60 rounds.  Warm conditions start at round 31
from the cold_fed round-30 checkpoint, so warm_{fed,local} are plotted from
round 31 only.

Panel A: Mean AUROC learning curves (all 4 conditions).
         Shading = +/- 1 seed-SD across 5 seeds.

Panel B: Min AUROC (worst-off provider) learning curves.
         Captures whether the tail of the provider distribution benefits from
         federation.

Panel C: warm_fed vs warm_local at common checkpoints (rounds 31-60).
         The gap isolates continued federation vs one-shot warm initialisation.
         Error bars = 1 SEM across seeds.

Data source: trust-ct/processed/e6_fairness/e6_summary_round_results.csv
"""

import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT   = pathlib.Path(__file__).resolve().parent.parent.parent
E6DIR  = ROOT / "trust-ct" / "processed" / "e6_fairness"
OUT    = os.path.join(os.path.dirname(__file__), "..", "figures", "fig6_e6_fairness.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_GRAY   = "#888888"

# -- Load summary CSV ----------------------------------------------------------
summ = pd.read_csv(E6DIR / "e6_summary_round_results.csv")

def seed_agg(summ, metric):
    g = summ.groupby(["condition", "round"])[metric]
    mu  = g.mean().rename("mu")
    sd  = g.std(ddof=1).rename("sd")
    sem = (g.std(ddof=1) / np.sqrt(g.count())).rename("sem")
    return pd.concat([mu, sd, sem], axis=1).reset_index()

mean_df = seed_agg(summ, "mean_auroc")
min_df  = seed_agg(summ, "min_auroc")

CONDITIONS = {
    "cold_fed":   ("FedAvg (cold start)",            C_BLUE,   "-",  "o"),
    "cold_local": ("Local-only (cold start)",         C_ORANGE, "--", "s"),
    "warm_fed":   ("FedAvg (warm init, cont.)",       C_GREEN,  "-",  "^"),
    "warm_local": ("Local-only (warm init, indep.)",  C_RED,    ":",  "D"),
}

# -- Figure layout -------------------------------------------------------------
fig = plt.figure(figsize=(16, 7.5))
gs  = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.2, 1.0], wspace=0.42)

# ======================================================================
# Panel A: Mean AUROC learning curves
# ======================================================================
ax_a = fig.add_subplot(gs[0])

for cond, (label, col, ls, mk) in CONDITIONS.items():
    sub = mean_df[mean_df.condition == cond].sort_values("round")
    if len(sub) == 0:
        continue
    ax_a.plot(sub["round"], sub["mu"],
              color=col, ls=ls, lw=2.0, marker=mk, ms=3,
              markevery=10, label=label, zorder=4)
    ax_a.fill_between(sub["round"],
                      sub["mu"] - sub["sd"],
                      sub["mu"] + sub["sd"],
                      color=col, alpha=0.10)

ax_a.axvline(30, color=C_GRAY, lw=0.9, ls=":", alpha=0.8,
             label="Warm-start checkpoint (r=30)")
ax_a.set_xlabel("Round", fontsize=11)
ax_a.set_ylabel("Mean AUROC across K=23 providers", fontsize=11)
ax_a.set_xlim(1, 60)
ax_a.set_title("(A) Mean AUROC learning curves\n"
               "Cimas, K=23 providers, 5 seeds",
               fontsize=12)
ax_a.tick_params(labelsize=10)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.legend(fontsize=9.5, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")

r60 = mean_df[mean_df["round"] == 60].set_index("condition")["mu"]
for cond, (_, col, _, _) in CONDITIONS.items():
    if cond in r60.index:
        ax_a.annotate(f"{r60[cond]:.4f}",
                      xy=(60, r60[cond]), xytext=(3, 0),
                      textcoords="offset points", fontsize=9,
                      color=col, va="center")

# ======================================================================
# Panel B: Min AUROC (worst-off provider) learning curves
# ======================================================================
ax_b = fig.add_subplot(gs[1])

for cond, (label, col, ls, mk) in CONDITIONS.items():
    sub = min_df[min_df.condition == cond].sort_values("round")
    if len(sub) == 0:
        continue
    ax_b.plot(sub["round"], sub["mu"],
              color=col, ls=ls, lw=2.0, marker=mk, ms=3,
              markevery=10, label=label, zorder=4)
    ax_b.fill_between(sub["round"],
                      sub["mu"] - sub["sd"],
                      sub["mu"] + sub["sd"],
                      color=col, alpha=0.10)

ax_b.axvline(30, color=C_GRAY, lw=0.9, ls=":", alpha=0.8)
ax_b.set_xlabel("Round", fontsize=11)
ax_b.set_ylabel("Min AUROC (worst-off provider)", fontsize=11)
ax_b.set_xlim(1, 60)
ax_b.set_title("(B) Worst-off provider AUROC\n"
               "Cimas, K=23 providers, 5 seeds",
               fontsize=12)
ax_b.tick_params(labelsize=10)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

r60_min = min_df[min_df["round"] == 60].set_index("condition")["mu"]
for cond, (_, col, _, _) in CONDITIONS.items():
    if cond in r60_min.index:
        ax_b.annotate(f"{r60_min[cond]:.4f}",
                      xy=(60, r60_min[cond]), xytext=(3, 0),
                      textcoords="offset points", fontsize=9,
                      color=col, va="center")

# ======================================================================
# Panel C: contrast forest plot (warm-local minus warm-FedAvg, locked values)
# ======================================================================
ax_c = fig.add_subplot(gs[2])

# Locked contrasts: warm-local minus warm-FedAvg
# Positive = warm-local higher (warm init alone sufficient)
# Negative = warm-FedAvg higher (continued federation benefits that group)
CONTRASTS = [
    ("Macro mean",         +0.00236, +0.00022, +0.00564, 0.213),
    ("Weighted mean",      +0.00014, -0.00090, +0.00153, 0.858),
    ("Worst provider",     -0.08669, -0.14415, -0.04118, 0.0457),
    ("p10 provider",       -0.02281, -0.03307, -0.01255, 0.0165),
]

y_pos = np.arange(len(CONTRASTS))[::-1]
SIG_ALPHA = 0.05

for i, (label_c, est, lo, hi, pval) in enumerate(CONTRASTS):
    y = y_pos[i]
    sig = pval < SIG_ALPHA
    col = C_RED if est < 0 else C_BLUE
    lw  = 2.2 if sig else 1.5
    ax_c.plot([lo, hi], [y, y], color=col, lw=lw, solid_capstyle="round", zorder=3)
    ax_c.plot([lo, lo], [y - 0.12, y + 0.12], color=col, lw=1.4)
    ax_c.plot([hi, hi], [y - 0.12, y + 0.12], color=col, lw=1.4)
    ax_c.plot(est, y, "o", color=col, ms=7, zorder=5)
    ax_c.text(hi + 0.002, y,
              f" {est:+.5f}\n [{lo:+.5f}, {hi:+.5f}]\n p={pval:.3f}",
              va="center", fontsize=9, color="#333333", fontfamily="monospace")

ax_c.axvline(0.0, color=C_GRAY, lw=1.0, ls="--", alpha=0.9)
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels([c[0] for c in CONTRASTS], fontsize=10)
ax_c.set_xlabel("warm-local minus warm-FedAvg", fontsize=11)
ax_c.set_title("(C) Warm-local vs warm-FedAvg contrasts\n"
               r"rounds 31–60, $K=23$ providers",
               fontsize=12)
ax_c.tick_params(labelsize=10)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.text(0.02, 0.04,
          "Exploratory, unadjusted p-values (no multiplicity correction).",
          transform=ax_c.transAxes, fontsize=8.5, color="#666666", va="bottom")

# -- Caption -------------------------------------------------------------------
fig.text(
    0.5, -0.03,
    "Panel A/B: shading = ±1 seed-SD (5 seeds, K=23 Cimas providers); "
    "warm conditions initialised from cold_fed round-30 checkpoint. "
    "The identity of the worst-off provider (Panel B) may vary across conditions and seeds. "
    "Panel C: contrasts are warm-local minus warm-FedAvg (positive = warm-local higher); "
    "error bars are 95% seed-level bootstrap confidence intervals; "
    r"$p$-values are from paired seed-level $t$-tests across five seeds (exploratory, unadjusted). "
    "Warm initialization explained most aggregate AUROC parity; "
    "continued federation retained advantages for lower-performing providers.",
    ha="center", va="top", fontsize=8.5, color="#555555",
)

plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
