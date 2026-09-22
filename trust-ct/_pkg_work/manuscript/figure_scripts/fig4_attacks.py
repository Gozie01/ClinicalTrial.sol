"""Figure 4 -- Adversarial robustness: attack degradation and defence comparison.

Data sources:
  - Original attacks (sign_flip, scaled_poison, model_replace, label_flip, alie):
      trust-ct/processed/phase6/phase6b_adversarial.csv
  - W09-corrected attacks (random_gauss, backdoor):
      trust-ct/processed/phase6/phase6b_adversarial_w09.csv

Panel A: Mean delta AUROC (attack AUROC - clean AUROC) for all 7 attack families
         at eICU, FedAvg, f=3 Byzantine providers.

Panel B: Mean AUROC under label_flip by defence strategy at eICU, f=3.
         Krum is inapplicable for eICU (K=8, f=3: requires K > 2f+2=8, but 8 > 8
         is False). Krum rows marked N/A; bar replaced with hatched indicator.

Removed from prior version:
  - Paper-bundle pre-W09 source (security_attacks_by_seed.csv).
  - Hardcoded p-values and stale clean AUROC of 0.908.

Standalone fix (2026-08-29): dual-layout path resolution; Krum N/A for eICU.
"""
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT  = pathlib.Path(__file__).resolve().parent.parent.parent
OUT   = os.path.join(os.path.dirname(__file__), "..", "figures", "fig4_attacks.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

_ARCHIVE_MAP = [
    (("cimas",    "results_phase3_freeze"), ("results", "phase3")),
    (("better_bp","results_phase4"),        ("results", "phase4")),
    (("phase5",   "results_v3"),            ("results", "phase5")),
    (("phase6",),                           ("results", "phase6")),
    (("ablation",),                         ("results", "ablation")),
]


def _locate(*parts):
    """Try live layout (ROOT/trust-ct/processed/...) then archive (ROOT/results/...)."""
    live = ROOT / "trust-ct" / "processed"
    for p in parts:
        live = live / p
    if live.exists():
        return live
    for prefix, (arc_base, arc_dir) in _ARCHIVE_MAP:
        n = len(prefix)
        if parts[:n] == prefix:
            alt = ROOT / arc_base / arc_dir
            for r in parts[n:]:
                alt = alt / r
            return alt
    raise FileNotFoundError(f"Cannot locate {parts!r}; live path checked: {live}")


P6DIR = _locate("phase6")

# Locked values (provenance: phase6b bootstrap, 2 000 resamples)
LOCK_CLEAN_AUROC      = 0.8904
LOCK_CLEAN_AUROC_CI   = (0.8892, 0.8913)
LOCK_LABFLIP_DEG      = (-0.120, -0.178, -0.061)   # (mean, lo95, hi95)
LOCK_CM_RECOVERY      = (+0.111, +0.058, +0.165)    # (mean, lo95, hi95)

C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY   = "#888888"

# -- Load and merge -----------------------------------------------------------
orig = pd.read_csv(P6DIR / "phase6b_adversarial.csv")
w09  = pd.read_csv(P6DIR / "phase6b_adversarial_w09.csv")

W09_ATTACKS  = {"random_gauss", "backdoor"}
orig_trimmed = orig[~orig["attack"].isin(W09_ATTACKS)].copy()

# Per-seed clean AUROC for computing degradation
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

# -- Panel A data -------------------------------------------------------------
attack_order = ["label_flip", "sign_flip", "random_gauss", "scaled_poison",
                "model_replace", "alie", "backdoor"]
attack_labels = {
    "label_flip":    "Label flip",
    "sign_flip":     "Sign flip",
    "random_gauss":  "Random Gaussian",
    "scaled_poison": "Scaled poison",
    "model_replace": "Model replace",
    "alie":          "ALIE",
    "backdoor":      "Backdoor",
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

bar_colors = [C_RED] * len(attack_order)

# -- Panel B data -------------------------------------------------------------
# Krum applicability check: eICU K=8, f=3 => requires K > 2f+2 = 8; 8 > 8 is False
defenses_b  = ["fedavg", "coord_median", "trimmed_mean", "krum", "clipping"]
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

# Krum N/A: eICU K=8, f=3 requires K > 2f+2 = 8; 8 > 8 is False — always N/A
krum_na = (DATASET == "eicu" and F_MAIN == 3)
if not krum_na:
    krum_rows = pb[pb.defense == "krum"]
    if len(krum_rows) > 0:
        if "krum_applicable" in krum_rows.columns:
            krum_na = not bool(krum_rows["krum_applicable"].any())
        elif "run_status" in krum_rows.columns:
            krum_na = (krum_rows["run_status"] == "not_applicable_diagnostic_fallback").all()

def_auroc = {}
for d in defenses_b:
    if d == "krum" and krum_na:
        def_auroc[d] = (np.nan, np.nan)
        continue
    sub = pb[pb.defense == d]["auroc"]
    def_auroc[d] = (float(sub.mean()), float(sub.std(ddof=1) / np.sqrt(len(sub)))) if len(sub) else (np.nan, np.nan)

means_b = np.array([def_auroc[d][0] for d in defenses_b])
sems_b  = np.array([def_auroc[d][1] for d in defenses_b])

# -- Figure layout ------------------------------------------------------------
fig = plt.figure(figsize=(12, 5.2))
gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1.0], wspace=0.40)

# -- Panel A ------------------------------------------------------------------
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
ax_a.set_title(
    f"(A) Attack-family utility degradation\n"
    f"eICU-CRD Demo, FedAvg, $f=3$ of $K=8$ Byzantine clients, "
    f"clean AUROC = {LOCK_CLEAN_AUROC:.4f} [{LOCK_CLEAN_AUROC_CI[0]:.4f}, {LOCK_CLEAN_AUROC_CI[1]:.4f}]",
    fontsize=9.0)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

legend_patches = [
    plt.Rectangle((0, 0), 1, 1, fc=C_RED + "BB", ec=C_RED, label="Attack families (7)"),
]
ax_a.legend(handles=legend_patches, fontsize=7.5, loc="upper right",
            framealpha=0.9, edgecolor="#CCCCCC")
ax_a.set_ylim(-0.22, 0.05)

for xi, (m, sem) in enumerate(zip(means_a, sems_a)):
    if not np.isnan(m):
        va = "top" if m < 0 else "bottom"
        offset = -0.004 if m < 0 else 0.004
        ax_a.text(xi, m + offset, f"{m:+.3f}", ha="center", va=va,
                  fontsize=7.0, color="#333333")

# Locked bootstrap CI annotation for label_flip
lf_xi = attack_order.index("label_flip")
ax_a.text(lf_xi, LOCK_LABFLIP_DEG[0] - 0.018,
          f"{LOCK_LABFLIP_DEG[0]:+.3f}\n[{LOCK_LABFLIP_DEG[1]:+.3f}, {LOCK_LABFLIP_DEG[2]:+.3f}]",
          ha="center", va="top", fontsize=7.0, color=C_RED,
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_RED, alpha=0.85))

# -- Panel B ------------------------------------------------------------------
ax_b = fig.add_subplot(gs[1])
xb = np.arange(len(defenses_b))

valid_means = [m for m in means_b if not np.isnan(m)]
y_lo = min(min(valid_means) - 0.04, CLEAN_AUROC - 0.02) if valid_means else CLEAN_AUROC - 0.06
y_hi = max(max(valid_means) + 0.03, CLEAN_AUROC + 0.02) if valid_means else CLEAN_AUROC + 0.06

def_colors = [C_RED, C_BLUE, C_GREEN, C_ORANGE, C_PURPLE]

for xi, (d, m, sem, col) in enumerate(zip(defenses_b, means_b, sems_b, def_colors)):
    if d == "krum" and krum_na:
        ax_b.text(xi, (y_lo + y_hi) / 2,
                  "N/A:\n$K=8\\leq 2f+2$",
                  ha="center", va="center", fontsize=7.5, color="#888888",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#F8F8F8",
                            ec="#CCCCCC", lw=0.8))
    else:
        ax_b.bar(xi, m, yerr=sem,
                 color=col + "BB", edgecolor=col, linewidth=1.2, width=0.55,
                 error_kw=dict(ecolor="#333333", capsize=3.5, lw=1.1), zorder=3)
        if not np.isnan(m):
            ax_b.text(xi, m + 0.004, f"{m:.4f}", ha="center", va="bottom",
                      fontsize=7.5, color="#333333")

# Locked coord-median recovery annotation
fedavg_idx = defenses_b.index("fedavg")
cm_idx     = defenses_b.index("coord_median")
fedavg_m   = means_b[fedavg_idx]
cm_m       = means_b[cm_idx]
if not (np.isnan(fedavg_m) or np.isnan(cm_m)):
    mid_y = (fedavg_m + cm_m) / 2
    ax_b.annotate("", xy=(cm_idx, cm_m), xytext=(cm_idx, fedavg_m),
                  arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=1.2))
    ax_b.text(cm_idx + 0.32, mid_y,
              f"recovery\n{LOCK_CM_RECOVERY[0]:+.3f}\n[{LOCK_CM_RECOVERY[1]:+.3f}, {LOCK_CM_RECOVERY[2]:+.3f}]",
              ha="left", va="center", fontsize=7.0, color=C_GREEN)

ax_b.axhline(CLEAN_AUROC, color=C_GRAY, lw=1.0, ls="--",
             label=f"Clean = {LOCK_CLEAN_AUROC:.4f}")
ax_b.set_xticks(xb)
ax_b.set_xticklabels([def_labels[d] for d in defenses_b], fontsize=8.5)
ax_b.set_ylabel("Mean AUROC (5 seeds)", fontsize=9)
ax_b.set_title("(B) Defence comparison under label-flip\n"
               r"eICU-CRD Demo, $f=3$ of $K=8$ Byzantine clients",
               fontsize=9.0)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.legend(fontsize=8, loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")
ax_b.set_ylim(y_lo, y_hi)


fig.text(0.5, -0.03,
         r"Panel A: mean $\pm$ 1 SEM over 5 seeds, eICU-CRD Demo, FedAvg aggregator. "
         r"Panel B: label-flip attack, $f=3$ of $K=8$ Byzantine clients, matched seeds.",
         ha="center", va="top", fontsize=7.5, color="#555555")

plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
