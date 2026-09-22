"""Figure 3 — Phase 4 policy contrasts and Phase 5 calibrated participation estimates.

Panel A: Phase 4 policy contrasts from policy_contrasts.csv (all at B=2/3).
         Point estimates only from policy_values.csv; individual policy CIs not stored.
         Primary contrast (uplift vs random) shown with real bootstrap CI.

Panel B: Phase 5 calibrated participation-rate estimates (C0-C3) with recorded 95% CIs
         from phase5_v3_lock.json betterbp_participation_ci.
         Phase 4 DR targets shown as reference marks.

Removed from prior version:
  - Reconstructed exponential learning curves (fabricated from endpoint targets).
  - Fabricated absolute CIs on individual policy values.
  - nAULC materiality band.
"""
import os
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT  = pathlib.Path(__file__).resolve().parent.parent.parent
P4DIR = ROOT / "trust-ct" / "processed" / "better_bp" / "results_phase4"
P5LK  = ROOT / "trust-ct" / "processed" / "phase5" / "results_v3" / "phase5_v3_lock.json"
OUT   = os.path.join(os.path.dirname(__file__), "..", "figures", "fig3_policy_replay.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_GRAY   = "#888888"
C_PURPLE = "#CC79A7"

# ── Load data ──────────────────────────────────────────────────────────────────
pv = pd.read_csv(P4DIR / "policy_values.csv")
pc = pd.read_csv(P4DIR / "policy_contrasts.csv")
p5 = json.loads(P5LK.read_text())

pv_b = pv[pv.budget == 0.6667].set_index("policy")
pc_b = pc[pc.budget == 0.6667]

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5.2))
gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1.0], wspace=0.44)


# ══════════════════════════════════════════════════════════════════════════════
# Panel A — Policy value point estimates + real contrast annotations
# (No fabricated individual-policy CIs; CI only on the contrast)
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])

policy_order  = ["no_incentive", "random_budget", "risk_targeting", "uplift_targeting"]
policy_labels = [r"$\pi_0$" + "\n(no incentive)",
                 r"$\pi_\mathrm{rand}$" + "\n(random budget)",
                 r"$\pi_\mathrm{risk}$" + "\n(risk-stratified)",
                 r"$\pi_\mathrm{up}$"  + "\n(uplift-targeting)"]
vals   = np.array([pv_b.loc[p, "dr_value"] for p in policy_order])
colors = [C_GRAY, C_ORANGE, C_BLUE, C_GREEN]

x = np.arange(len(policy_order))
bars = ax_a.bar(x, vals,
                color=[c + "AA" for c in colors],
                edgecolor=colors, linewidth=1.4, width=0.55, zorder=3)

# Random budget reference line
ax_a.axhline(vals[1], color=C_ORANGE, ls="--", lw=1.1, alpha=0.7,
             label=r"$\hat{{V}}(\pi_\mathrm{{rand}})={v:.3f}$".format(v=vals[1]))

# Primary contrast annotation: uplift vs random (real CI from policy_contrasts.csv)
pc_primary = pc_b[(pc_b.policy_a == "uplift_targeting") &
                  (pc_b.policy_b == "random_budget") &
                  (pc_b.role     == "primary")].iloc[0]
dv  = pc_primary["delta_v"]
lcb = pc_primary["lcb_95"]
ucb = pc_primary["ucb_95"]

y_ann = vals[3] + 0.008
ax_a.annotate("", xy=(3, vals[3]), xytext=(3, vals[1]),
              arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.2))
ax_a.text(3.30, (vals[3] + vals[1]) / 2,
          f"$\\Delta={dv:+.3f}$\n$[{lcb:+.3f},\\ {ucb:+.3f}]$\n(CI crosses 0)",
          ha="left", va="center", fontsize=7.5, color=C_RED)

# Value labels on bars
for xi, v in enumerate(vals):
    ax_a.text(xi, v + 0.003, f"{v:.4f}", ha="center", va="bottom",
              fontsize=7.5, color="#333333")

ax_a.set_xticks(x)
ax_a.set_xticklabels(policy_labels, fontsize=8.5)
ax_a.set_ylabel(r"Estimated policy value $\hat{V}(\pi)$", fontsize=9)
ax_a.set_ylim(0.78, 0.955)
ax_a.set_title("(A) Phase 4 policy-value estimates\n"
               r"BETTER-BP, $N=402$, $B=2/3$, $N_\mathrm{boot}=2{,}000$",
               fontsize=9.5)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.legend(fontsize=8, loc="upper left", framealpha=0.85, edgecolor="#CCCCCC")
ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
ax_a.text(0.02, 0.02,
          "Individual policy CIs not stored.\nContrast CI from policy_contrasts.csv.",
          transform=ax_a.transAxes, fontsize=7, color="#888888", va="bottom")


# ══════════════════════════════════════════════════════════════════════════════
# Panel B — Phase 5 calibrated participation-rate estimates (C0-C3)
# Source: phase5_v3_lock.json betterbp_participation_ci
# Phase 4 DR targets shown as reference diamonds
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1])

p5_ci  = p5["betterbp_participation_ci"]
scenarios   = ["C0", "C1", "C2", "C3"]
scen_labels = ["C0\n(control)", "C1\n(random)", "C2\n(risk)", "C3\n(uplift)"]
scen_colors = [C_GRAY, C_ORANGE, C_BLUE, C_GREEN]

p5_means = np.array([p5_ci[s]["mean"]   for s in scenarios])
p5_lo    = np.array([p5_ci[s]["lcb95"]  for s in scenarios])
p5_hi    = np.array([p5_ci[s]["ucb95"]  for s in scenarios])
p4_tgt   = np.array([p5_ci[s]["phase4_dr_target"] for s in scenarios])

y_pos = np.arange(len(scenarios))

for i, (s, m, lo, hi, col, tgt) in enumerate(
        zip(scenarios, p5_means, p5_lo, p5_hi, scen_colors, p4_tgt)):
    # CI bar
    ax_b.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round", zorder=3)
    ax_b.plot([lo, lo], [i - 0.12, i + 0.12], color=col, lw=1.5)
    ax_b.plot([hi, hi], [i - 0.12, i + 0.12], color=col, lw=1.5)
    # Point estimate
    ax_b.plot(m, i, "o", color=col, ms=8, zorder=5, label=scen_labels[i])
    # Phase 4 DR target reference
    ax_b.plot(tgt, i, "D", color=col, ms=6, mfc="white", mew=1.5, zorder=4)
    # Annotation
    ax_b.text(hi + 0.0005, i,
              f" {m:.4f} [{lo:.4f}, {hi:.4f}]",
              va="center", fontsize=7.2, color="#333333", fontfamily="monospace")

ax_b.axvline(p5_means[0], color=C_GRAY, lw=0.8, ls=":", alpha=0.6)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(scen_labels, fontsize=9.5)
ax_b.set_xlabel("Calibrated participation-rate estimate", fontsize=9)
ax_b.set_xlim(0.805, 0.910)
ax_b.set_title("(B) Phase 5 calibrated participation estimates\n"
               r"BETTER-BP replay simulation, $N_\mathrm{boot}=500$",
               fontsize=9.5)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

legend_elems = [
    plt.Line2D([0], [0], color="#555555", lw=2, marker="o", ms=7,
               label="Phase 5 estimate [95% CI]"),
    plt.Line2D([0], [0], color="#555555", lw=0, marker="D", ms=6,
               mfc="white", mew=1.5, label="Phase 4 DR target (reference)"),
]
ax_b.legend(handles=legend_elems, fontsize=8, loc="lower right",
            framealpha=0.9, edgecolor="#CCCCCC")

fig.text(0.5, -0.03,
         "Panel A: policy-value point estimates from policy_values.csv; "
         "contrast CI from policy_contrasts.csv (N_boot=2,000, seeds=[7,11,19,23,37]).\n"
         "Panel B: participation estimates from phase5_v3_lock.json "
         "betterbp_participation_ci; Phase 4 DR targets shown as open diamonds.",
         ha="center", va="top", fontsize=7.5, color="#555555")

plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
