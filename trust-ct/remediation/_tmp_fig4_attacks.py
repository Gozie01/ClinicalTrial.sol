"""Figure 4 — Adversarial robustness: attack degradation and defence comparison.

Data sources:
  - Original attacks (sign_flip, scaled_poison, model_replace, label_flip, alie):
      trust-ct/processed/phase6/phase6b_adversarial.csv
  - W09-corrected attacks (random_gauss, backdoor):
      trust-ct/processed/phase6/phase6b_adversarial_w09.csv

Panel A: Mean delta AUROC (attack AUROC - clean AUROC) for all 7 attack families
         at eICU, FedAvg, f=3 Byzantine providers.
         Clean AUROC = mean over 5 seeds (attack=none, f=0, defense=fedavg, dataset=eicu).

Panel B: Mean AUROC under label_flip by defence strategy
         at eICU, f=3.  Compared across 5 matched seeds.

Removed from prior version:
  - Paper-bundle pre-W09 source (security_attacks_by_seed.csv).
  - Hardcoded p-values and stale clean AUROC of 0.908.
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
P6DIR  = ROOT / "trust-ct" / "processed" / "phase6"
OUT    = os.path.join(os.path.dirname(__file__), "..", "figures", "fig4_attacks.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY   = "#888888"

# ── Load and merge ──────────────────────────────────────────────────────────────
orig = pd.read_csv(P6DIR / "phase6b_adversarial.csv")
w09  = pd.read_csv(P6DIR / "phase6b_adversarial_w09.csv")

W09_ATTACKS = {"random_gauss", "backdoor"}
orig_trimmed = orig[~orig["attack"].isin(W09_ATTACKS)].copy()

# Per-seed clean AUROC (attack=none, fedavg, eicu, f=0) for computing degradation
clean_seed = (orig[(orig.attack == "none") & (orig.defense == "fedavg") &
                   (orig.dataset == "eicu") & (orig.f == 0)]
              .set_index("seed")["auroc"])

w09 = w09.copy()
w09["auroc_clean"]       = w09["seed"].map(clean_seed)
w09["auroc_degradation"] = w09["auroc"] - w09["auroc_clean"]

if "auroc_degradation" not in orig_trimmed.columns:
    orig_trimmed["auroc_degradation"] = orig_trimmed["auroc"] - orig_trimmed["auroc_clean"]

shared_cols = ["dataset", "attack", "f", "defense", "seed",
               "auroc", "auroc_clean", "auroc_degradation"]
merged = pd.concat([orig_trimmed[shared_cols], w09[shared_cols]], ignore_index=True)

DATASET     = "eicu"
DEFENSE     = "fedavg"
F_MAIN      = 3
CLEAN_AUROC = float(clean_seed.mean())

# ── Panel A data ───────────────────────────────────────────────────────────────
attack_order = ["label_flip", "sign_flip", "random_gauss", "scaled_poison",
                "model_replace", "alie", "backdoor"]
attack_labels = {
    "label_flip":    "Label flip",
    "sign_flip":     "Sign flip",
    "random_gauss":  "Random Gaussian\n(W09-corrected)",
    "scaled_poison": "Scaled poison",
    "model_replace": "Model replace",
    "alie":          "ALIE",
    "backdoor":      "Backdoor\n(W09-corrected)",
}

pa = merged[(merged.dataset == DATASET) &
            (merged.defense == DEFENSE) &
            (merged.f == F_MAIN)].copy()

atk_deg = {}
for atk in attack_order:
    sub = pa[pa.attack == atk]["auroc_degradation"]
    atk_deg[atk] = (float(sub.mean()), float(sub.std(ddof=1) / np.sqrt(len(sub)))) if len(sub) else (np.nan, np.nan)

means_a = np.array([atk_deg[a][0] for a in attack_order])
sems_a  = np.array([atk_deg[a][1] for a in attack_order])

bar_colors = []
for atk in attack_order:
    if atk in W09_ATTACKS:
        bar_colors.append(C_ORANGE)
    elif atk == "alie":
        bar_colors.append(C_PURPLE)
    else:
        bar_colors.append(C_RED)

# ── Panel B data ───────────────────────────────────────────────────────────────
defenses_b = ["fedavg", "coord_median", "trimmed_mean", "krum", "clipping"]
def_labels  = {
    "fedavg":       "FedAvg\n(no defence)",
    "coord_median": "Coord.\nmedian",
    "trimmed_mean": "Trimmed\nmean",
    "krum":         "Krum",
    "clipping":     "Clipping",
}
pb = merged[(merged.dataset == DATASET) &
            (merged.attack  == "label_flip") &
            (merged.f       == F_MAIN)].copy()

def_auroc = {}
for d in defenses_b:
    sub = pb[pb.defense == d]["auroc"]
    def_auroc[d] = (float(sub.mean()), float(sub.std(ddof=1) / np.sqrt(len(sub)))) if len(sub) else (np.nan, np.nan)

means_b = np.array([def_auroc[d][0] for d in defenses_b])
sems_b  = np.array([def_auroc[d][1] for d in defenses_b])

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5.2))
gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1.0], wspace=0.40)

# ── Panel A ─────────────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0])
xa = np.arange(len(attack_order))
ax_a.bar(xa, means_a, yerr=sems_a,
         color=[c + "BB" for c in bar_colors],
         edgecolor=bar_colors, linewidth=1.2, width=0.58,
         error_kw=dict(ecolor="#333333", capsize=3.5, lw=1.1), zorder=3)

ax_a.axhline(0.0, color=C_GRAY, lw=0.9, ls=":")
ax_a.set_xticks(xa)
ax_a.set_xticklabels([attack_labels[a] for a in attack_order],
                     rotation=28, ha="right", fontsize=8.0)
ax_a.set_ylabel(r"$\Delta$AUROC  (attack $-$ clean)", fontsize=9)
ax_a.set_title(f"(A) Attack-family utility degradation\n"
               f"eICU, FedAvg, $f=3$ Byzantine, clean AUROC = {CLEAN_AUROC:.4f}",
               fontsize=9.5)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

legend_patches = [
    plt.Rectangle((0, 0), 1, 1, fc=C_RED    + "BB", ec=C_RED,    label="Original (phase6b_adversarial.csv)"),
    plt.Rectangle((0, 0), 1, 1, fc=C_ORANGE + "BB", ec=C_ORANGE, label="W09-corrected (random_gauss, backdoor)"),
    plt.Rectangle((0, 0), 1, 1, fc=C_PURPLE + "BB", ec=C_PURPLE, label="ALIE"),
]
ax_a.legend(handles=legend_patches, fontsize=7.5, loc="lower left",
            framealpha=0.9, edgecolor="#CCCCCC")

for xi, (m, sem) in enumerate(zip(means_a, sems_a)):
    if not np.isnan(m):
        va = "top" if m < 0 else "bottom"
        offset = -0.004 if m < 0 else 0.004
        ax_a.text(xi, m + offset, f"{m:+.3f}", ha="center", va=va,
                  fontsize=7.0, color="#333333")

# ── Panel B ─────────────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[1])
xb = np.arange(len(defenses_b))
def_colors = [C_RED, C_BLUE, C_GREEN, C_ORANGE, C_PURPLE]
ax_b.bar(xb, means_b, yerr=sems_b,
         color=[c + "BB" for c in def_colors],
         edgecolor=def_colors, linewidth=1.2, width=0.55,
         error_kw=dict(ecolor="#333333", capsize=3.5, lw=1.1), zorder=3)

ax_b.axhline(CLEAN_AUROC, color=C_GRAY, lw=1.0, ls="--",
             label=f"Clean AUROC = {CLEAN_AUROC:.4f}")
ax_b.set_xticks(xb)
ax_b.set_xticklabels([def_labels[d] for d in defenses_b], fontsize=8.5)
ax_b.set_ylabel("Mean AUROC (5 seeds)", fontsize=9)
ax_b.set_title("(B) Defence comparison under label\_flip\n"
               r"eICU, $f=3$ Byzantine",
               fontsize=9.5)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.legend(fontsize=8, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")

for xi, (m, sem) in enumerate(zip(means_b, sems_b)):
    if not np.isnan(m):
        ax_b.text(xi, m + 0.004, f"{m:.4f}", ha="center", va="bottom",
                  fontsize=7.5, color="#333333")

y_lo = min(float(np.nanmin(means_b)) - 0.04, CLEAN_AUROC - 0.02)
y_hi = max(float(np.nanmax(means_b)) + 0.03, CLEAN_AUROC + 0.02)
ax_b.set_ylim(y_lo, y_hi)

fig.text(0.5, -0.03,
         r"Panel A: mean $\pm$ 1 SEM over 5 seeds. "
         r"W09-corrected sources used for random\_gauss and backdoor. "
         r"Panel B: label\_flip, $f=3$, eICU, matched seeds.",
         ha="center", va="top", fontsize=7.5, color="#555555")

plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
