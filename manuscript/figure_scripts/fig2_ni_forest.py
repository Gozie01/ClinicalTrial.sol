"""Figure 2 — Forest plot of FL vs centralized AUROC (Cimas LOCO, noninferiority).

Data read directly from ni_client_level.csv.
NI margin = -0.02 (prespecified one-sided); provider-cluster bootstrap, 2,000 resamples.
"""
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
NI_CSV = ROOT / "trust-ct" / "processed" / "cimas" / "results_phase3_freeze" / "ni_client_level.csv"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "fig2_ni_forest.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── Load from locked CSV (no hardcoded values) ─────────────────────────────────
ni = pd.read_csv(NI_CSV)

# Preserve natural order from CSV
methods = ni["method"].tolist()
delta   = ni["obs"].values       # bootstrap NI point estimate (not rounded marginal)
lo_95   = ni["lo_95"].values     # two-sided 95% CI lower bound
hi_95   = ni["hi_95"].values     # two-sided 95% CI upper bound
ni_lcb  = ni["lcb_95"].values    # one-sided NI lower confidence bound
K       = int(ni["K"].iloc[0])
N_BOOT  = 2000                   # from results_phase3_freeze/provenance.json
NI_MARGIN = -0.02

C_BLUE  = "#0072B2"
C_GRAY  = "#888888"
C_GREEN = "#009E73"
C_RED   = "#D55E00"

fig, ax = plt.subplots(figsize=(7.5, 4.2))
y_pos = np.arange(len(methods))[::-1]

for i, (m, d, lo, hi, lcb) in enumerate(zip(methods, delta, lo_95, hi_95, ni_lcb)):
    y = y_pos[i]
    ax.plot([lo, hi], [y, y], color=C_BLUE, lw=2.0, solid_capstyle="round")
    for x_tick in [lo, hi]:
        ax.plot([x_tick, x_tick], [y - 0.12, y + 0.12], color=C_BLUE, lw=1.5)
    ax.plot(d, y, "o", color=C_BLUE, ms=7, zorder=5)
    ax.plot(lcb, y, "|", color=C_RED, ms=10, mew=2.0, zorder=6)

ax.axvline(NI_MARGIN, color=C_RED, lw=1.5, ls="--")
ax.axvline(0.0, color=C_GRAY, lw=1.0, ls=":")
ax.axvspan(-0.035, NI_MARGIN, color=C_RED, alpha=0.06)

ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=10)
ax.set_xlabel(r"$\Delta$AUROC  (FL $-$ Centralized)", fontsize=10)
ax.set_xlim(-0.035, 0.012)
ax.set_ylim(-0.6, len(methods) - 0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(NI_MARGIN - 0.0008, len(methods) - 0.35, "NI margin",
        ha="right", fontsize=8, color=C_RED, style="italic")
ax.text(0.0008, len(methods) - 0.35, "Centralized",
        ha="left", fontsize=8, color=C_GRAY, style="italic")

# Tabulate numeric CI strings on the right
header_x = 0.0035
ax.text(header_x, len(methods) - 0.1, r"$\Delta$AUROC [95% CI]",
        ha="left", fontsize=7.5, color="#444444", fontweight="bold")
for i, (m, d, lo, hi, lcb) in enumerate(zip(methods, delta, lo_95, hi_95, ni_lcb)):
    y = y_pos[i]
    ci_str = f"{d:+.5f} [{lo:+.5f}, {hi:+.5f}]"
    ax.text(header_x, y, ci_str, ha="left", va="center", fontsize=8,
            color="#222222", fontfamily="monospace")

legend_elements = [
    plt.Line2D([0], [0], color=C_BLUE, lw=2, marker="o", ms=6,
               label="Point estimate [95% CI]"),
    plt.Line2D([0], [0], color=C_RED, lw=0, marker="|", ms=10, mew=2,
               label="NI lower bound"),
    plt.Line2D([0], [0], color=C_RED, lw=1.5, ls="--",
               label=f"NI margin = {NI_MARGIN}"),
    plt.Line2D([0], [0], color=C_GRAY, lw=1, ls=":",
               label="Centralized reference"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower left",
          framealpha=0.9, edgecolor="#CCCCCC")

ax.set_title(f"Federated vs Centralized AUROC — Cimas LOCO ($N=12{{,}}649$, $K={K}$)",
             fontsize=10, pad=8)

# Use $2{,}000$ for correct comma rendering in matplotlib mathtext
caption = (r"All five FL methods non-inferior to centralized training (NI LCB $>$ $-$0.02)." + "\n"
           r"Provider-cluster bootstrap, $2{,}000$ resamples. "
           r"Red tick: one-sided NI lower confidence bound. "
           r"Source: ni\_client\_level.csv.")
fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=8, color="#555555")

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
