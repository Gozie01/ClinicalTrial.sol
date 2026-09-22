"""Figure 5 -- Privacy-utility frontier (three-panel).

Panel A: Mean AUROC vs perturbation multiplier alpha from phase6c_leakage_fl_w09.csv.
         Actual evaluated alpha grid: {0, 0.01, 0.05, 0.10, 0.20}.
         X-axis: categorical positions to prevent 0/0.01 overlap on linear scale.
         Selected utility-preserving operating point alpha=0.01 marked.

Panel B: Reconstruction cosine vs batch size from phase6c_reconstruction.csv.
         Lines per alpha; batch_size on x-axis.

Panel C: Membership-inference AUROC (eICU and Cimas) at single evaluated setting
         (alpha=0 / no perturbation, 5-seed bootstrap).

Terminology: "perturbation multiplier alpha" replaces "DP noise level alpha".
             Clipped-Gaussian mechanism is not described as differential privacy.

Removed from prior version:
  - Paper-bundle pre-W07 source (model_inversion_by_seed.csv).
  - Hardcoded / stale batch=1 cosines.
  - Per-alpha MIA trajectory (not in locked data).
  - Feature sign-recovery analysis (AUTHOR_PENDING).

Standalone fix (2026-08-29): dual-layout path resolution; DP terminology removed;
categorical x-axis; alpha=0.01 operating point marker.
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
OUT   = os.path.join(os.path.dirname(__file__), "..", "figures", "fig5_frontier.pdf")
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

# Locked utility changes (provenance: phase6c_leakage_fl_w09.csv, W09-corrected)
LOCK_ALPHA_01_CHANGE = -0.001   # approx change at alpha=0.01
LOCK_ALPHA_05_CHANGE = -0.011   # approx change at alpha=0.05

C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY   = "#888888"

# -- Load data ----------------------------------------------------------------
lk  = pd.read_csv(P6DIR / "phase6c_leakage_fl_w09.csv")
rec = pd.read_csv(P6DIR / "phase6c_reconstruction.csv")
mia = pd.read_csv(P6DIR / "phase6c_mia.csv")

# -- Panel A: AUROC vs perturbation multiplier (categorical x) ----------------
alpha_grid = sorted(lk["alpha"].unique())          # [0, 0.01, 0.05, 0.10, 0.20]
x_pos      = np.arange(len(alpha_grid))            # [0, 1, 2, 3, 4]
x_labels   = [str(a) for a in alpha_grid]          # ["0", "0.01", ...]
pa_mean    = lk.groupby("alpha")["auroc"].mean()
pa_sem     = lk.groupby("alpha")["auroc"].sem()

SELECTED_ALPHA = 0.01
sel_idx = alpha_grid.index(SELECTED_ALPHA)         # position 1

# -- Panel B: Reconstruction cosine vs batch size -----------------------------
rec_agg = rec.groupby(["alpha", "batch_size"])["cos_sim"].agg(["mean", "sem"]).reset_index()
rec_agg.columns = ["alpha", "batch_size", "cos_mean", "cos_sem"]
batch_sizes  = sorted(rec["batch_size"].unique())
alpha_lines  = sorted(rec["alpha"].unique())
alpha_colors = {0.00: C_GRAY, 0.01: C_BLUE, 0.05: C_ORANGE, 0.10: C_GREEN, 0.20: C_RED}
alpha_markers= {0.00: "o", 0.01: "s", 0.05: "^", 0.10: "D", 0.20: "v"}

# -- Panel C: MIA bootstrap CI ------------------------------------------------
def boot_ci(values, n_boot=2000, alpha_ci=0.05, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    boot = [np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boot, 100 * alpha_ci / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha_ci / 2)))
    return float(arr.mean()), lo, hi

mia_eicu  = mia[mia.dataset == "eicu"]["mia_auroc"].values
mia_cimas = mia[mia.dataset == "cimas"]["mia_auroc"].values
mia_eicu_m,  mia_eicu_lo,  mia_eicu_hi  = boot_ci(mia_eicu,  seed=42)
mia_cimas_m, mia_cimas_lo, mia_cimas_hi = boot_ci(mia_cimas, seed=43)

# -- Figure layout ------------------------------------------------------------
fig = plt.figure(figsize=(14, 5.0))
gs  = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.3, 0.9], wspace=0.42)

# -- Panel A ------------------------------------------------------------------
ax_a = fig.add_subplot(gs[0])
ax_a.errorbar(x_pos, pa_mean.loc[alpha_grid].values,
              yerr=pa_sem.loc[alpha_grid].values,
              fmt="o-", color=C_BLUE, lw=2.0, ms=6,
              capsize=4, elinewidth=1.1, zorder=4, label="FedAvg AUROC")

clean_auroc = float(pa_mean.loc[0.0])
ax_a.axhline(clean_auroc, color=C_GRAY, lw=1.0, ls=":", label=f"No-noise baseline ({clean_auroc:.4f})")
ax_a.axhline(clean_auroc - 0.02, color=C_RED, lw=0.9, ls="--",
             alpha=0.8, label="Materiality threshold (-0.02)")

# Mark selected operating point alpha=0.01 (categorical position 1)
ax_a.axvline(sel_idx, color=C_GREEN, lw=1.4, ls="--", alpha=0.9,
             label=f"Selected op. point (alpha={SELECTED_ALPHA})", zorder=3)
ax_a.annotate(f"alpha={SELECTED_ALPHA}\n(selected)",
              xy=(sel_idx, pa_mean.loc[SELECTED_ALPHA]),
              xytext=(sel_idx + 0.35, pa_mean.loc[SELECTED_ALPHA] + 0.008),
              fontsize=7, color=C_GREEN, ha="left",
              arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=0.9))

for xi, (a, m, s) in enumerate(zip(alpha_grid, pa_mean.loc[alpha_grid].values,
                                    pa_sem.loc[alpha_grid].values)):
    ax_a.annotate(f"{m:.4f}", (xi, m), textcoords="offset points",
                  xytext=(4, 5), fontsize=7.5, color=C_BLUE)

# Locked delta annotations for alpha=0.01 and alpha=0.05
for a_val, a_idx, dlbl in [(0.01, 1, f"≈{LOCK_ALPHA_01_CHANGE:+.3f}"),
                            (0.05, 2, f"≈{LOCK_ALPHA_05_CHANGE:+.3f}")]:
    m_a = float(pa_mean.loc[a_val])
    ax_a.annotate(dlbl,
                  xy=(a_idx, m_a),
                  xytext=(a_idx + 0.22, m_a - 0.012),
                  fontsize=7.5, color=C_ORANGE, ha="left",
                  arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.8))

ax_a.set_xlabel(r"Perturbation multiplier $\alpha$", fontsize=9)
ax_a.set_ylabel("Mean AUROC (eICU, 5 seeds)", fontsize=9)
ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(x_labels, fontsize=8)
ax_a.set_ylim(0.72, 0.915)
ax_a.set_title("(A) Utility vs clipped-Gaussian update-perturbation\n"
               r"eICU, FedAvg, 5 seeds — multiplier $\alpha$",
               fontsize=9.5)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.legend(fontsize=7.0, loc="lower left", framealpha=0.9, edgecolor="#CCCCCC")

# -- Panel B ------------------------------------------------------------------
ax_b = fig.add_subplot(gs[1])
for a in alpha_lines:
    sub = rec_agg[rec_agg.alpha == a].sort_values("batch_size")
    col = alpha_colors.get(a, "#333333")
    mk  = alpha_markers.get(a, "o")
    lbl = rf"$\alpha={a}$" + (" (selected)" if a == SELECTED_ALPHA else "")
    ax_b.errorbar(sub["batch_size"], sub["cos_mean"], yerr=sub["cos_sem"],
                  fmt=mk + "-", color=col, lw=1.6, ms=6, capsize=3,
                  elinewidth=1.0, label=lbl)

ax_b.set_xscale("log")
ax_b.set_xticks(batch_sizes)
ax_b.set_xticklabels([str(b) for b in batch_sizes], fontsize=9)
ax_b.set_xlabel("Batch size (log scale)", fontsize=9)
ax_b.set_ylabel("Reconstruction cosine similarity", fontsize=9)
ax_b.set_ylim(-0.12, 1.08)
ax_b.axhline(0.0, color=C_GRAY, lw=0.8, ls=":")
ax_b.set_title("(B) Reconstruction cosine similarity vs batch size\n"
               r"Clipped-Gaussian perturbation, eICU, 5 seeds",
               fontsize=9.5)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#CCCCCC",
            title=r"Perturbation ($\alpha$)", title_fontsize=7.5)

# -- Panel C ------------------------------------------------------------------
ax_c = fig.add_subplot(gs[2])

datasets  = ["eICU", "Cimas"]
mia_means = [mia_eicu_m,  mia_cimas_m]
mia_los   = [mia_eicu_lo, mia_cimas_lo]
mia_his   = [mia_eicu_hi, mia_cimas_hi]
cols_c    = [C_RED, C_BLUE]

for i, (ds, m, lo, hi, col) in enumerate(
        zip(datasets, mia_means, mia_los, mia_his, cols_c)):
    ax_c.plot([lo, hi], [i, i], color=col, lw=3.0, solid_capstyle="round", zorder=3)
    ax_c.plot([lo, lo], [i - 0.15, i + 0.15], color=col, lw=1.5)
    ax_c.plot([hi, hi], [i - 0.15, i + 0.15], color=col, lw=1.5)
    ax_c.plot(m, i, "o", color=col, ms=9, zorder=5)
    ax_c.text(hi + 0.0003, i,
              f" {m:.5f}\n [{lo:.5f}, {hi:.5f}]",
              va="center", fontsize=7.5, color="#333333", fontfamily="monospace")

ax_c.axvline(0.5, color=C_GRAY, lw=1.0, ls="--",
             label="Chance (0.50)", alpha=0.9)
ax_c.set_yticks([0, 1])
ax_c.set_yticklabels(datasets, fontsize=10.5)
ax_c.set_xlabel("MIA AUROC", fontsize=9)
ax_c.set_xlim(0.490, 0.518)
ax_c.set_title("(C) Membership-inference AUROC\n"
               r"$\alpha=0$ (no perturbation), 5-seed bootstrap",
               fontsize=9.5)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#CCCCCC")
ax_c.text(0.02, -0.22,
          r"Evaluated at $\alpha=0$ only." + "\nNo per-α MIA trajectory in locked data.",
          transform=ax_c.transAxes, fontsize=7, color="#888888", va="top")

fig.text(0.5, -0.03,
         r"Panel A: 5-seed mean AUROC vs clipped-Gaussian update-perturbation multiplier $\alpha$; "
         r"$\alpha=0.01$ selected operating point. "
         r"Panel B: gradient reconstruction cosine similarity vs batch size. "
         r"Panel C: membership-inference AUROC at $\alpha=0$; 5-seed 95% bootstrap CI.",
         ha="center", va="top", fontsize=7.5, color="#555555")

plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
