"""Figure 1 -- TRUST-CT System Workflow diagram (matplotlib-drawn).

Audit log: HMAC-SHA256 signature chain sₜ=HMAC-SHA256(key,dₜ); cₜ=H₆₄(cₜ₋₁‖dₜ‖sₜ).
Policy/Replay: calibrated participation estimates (C0-C3).
Equation: clip_C notation in mathrm form.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "fig1_workflow.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY   = "#BBBBBB"

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11)
ax.set_ylim(0, 7)
ax.axis("off")


def box(ax, x, y, w, h, label, sublabel="", color=C_BLUE, fontsize=8.5, lw=1.4):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          linewidth=lw, edgecolor=color,
                          facecolor=color + "22")
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.12, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color)
        ax.text(x + w/2, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=7.2, color="#444444")
    else:
        ax.text(x + w/2, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color)


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"))


def label(ax, x, y, txt, fs=7.5, color="#333333", ha="center"):
    ax.text(x, y, txt, ha=ha, va="center", fontsize=fs, color=color)


# -- Title -------------------------------------------------------------------
ax.text(5.5, 6.7, "TRUST-CT Closed-Loop Federated Learning Framework",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#222222")

# -- Cohort blocks ------------------------------------------------------------
cohorts = [
    ("Cimas\n(K=23 providers)",       "Adherence prediction\nN=12,649",      0.15, 4.8, C_BLUE),
    ("BETTER-BP\n(K=2 sites)",        "Policy eval. / participation\nN=402", 0.15, 3.0, C_GREEN),
    ("eICU-CRD Demo\n(K=8 hosp. groups)", "Security & privacy bench.\nN=2,520",  0.15, 1.2, C_RED),
]
for title, sub, x, y, col in cohorts:
    box(ax, x, y, 2.2, 1.4, title, sub, color=col)

for y_ann in [4.5, 2.7, 0.9]:
    label(ax, 1.25, y_ann, "Provider-partitioned\nvirtual clients", fs=7, color="#777777")

# -- Arrows: cohorts -> FL coordinator ----------------------------------------
for y_src in [5.5, 3.7, 1.9]:
    arrow(ax, 2.35, y_src, 3.5, y_src)

# -- FL Coordinator -----------------------------------------------------------
box(ax, 3.5, 2.4, 2.4, 3.5,
    "FL Coordinator",
    "FedAvg / FedProx / SCAFFOLD\nFedAvg-equal / FedAdam\nCoord. median | Trimmed mean",
    color=C_BLUE, fontsize=9)

# Clipping equation: clip_C notation (mathrm, no operatorname needed in mathtext)
ax.text(4.7, 2.55,
        r"$\tilde{\Delta}_k = \mathrm{clip}_C(\Delta_k) + \varepsilon,$"
        "\n"
        r"$\varepsilon \sim \mathcal{N}(0,(\alpha C)^2 I)$",
        ha="center", va="center", fontsize=7, color="#555555", style="italic")

# -- Arrows: coordinator -> modules -------------------------------------------
arrow(ax, 5.9, 5.1, 7.1, 5.5, color=C_ORANGE)
arrow(ax, 5.9, 3.6, 7.1, 3.6, color=C_PURPLE)
arrow(ax, 5.9, 2.6, 7.1, 1.7, color=C_GREEN)

# -- Policy Evaluation & Closed-Loop Replay -----------------------------------
box(ax, 7.1, 4.8, 3.0, 1.5,
    "Policy Evaluation\n& Closed-Loop Replay",
    "DR uplift | BETTER-BP\nCalibrated participation est.\n(C0-C3, 95% CI from simulation)",
    color=C_ORANGE, fontsize=8)

# -- Privacy & Security Evaluation --------------------------------------------
box(ax, 7.1, 2.85, 3.0, 1.6,
    "Privacy & Security Eval.",
    "Clipped-Gaussian perturbation\nRecon. cosine | MIA AUROC\nAttack families (7) | Defenses (5)",
    color=C_PURPLE, fontsize=8)

# -- HMAC-Authenticated Hash-Linked Audit Log ---------------------------------
box(ax, 7.1, 0.65, 3.0, 1.9,
    "Hash-Linked Audit Log",
    u"dₜ = H(canonical(eₜ))\n"
    u"sₜ = HMAC-SHA256(key, dₜ)\n"
    u"cₜ = H₆₄(cₜ₋₁ ‖ dₜ ‖ sₜ)\n"
    "350/350 fault injections detected",
    color=C_GREEN, fontsize=7.0)

# -- Bottom note --------------------------------------------------------------
ax.text(5.5, 0.18,
        "Raw clinical records remain at each site. "
        "No secure aggregation or distributed consensus implemented.",
        ha="center", va="center", fontsize=7.5, color="#666666", style="italic")

# -- Legend -------------------------------------------------------------------
patches = [
    mpatches.Patch(facecolor=C_BLUE   + "33", edgecolor=C_BLUE,   label="Federated prediction (Cimas)"),
    mpatches.Patch(facecolor=C_GREEN  + "33", edgecolor=C_GREEN,  label="Policy eval. / audit log"),
    mpatches.Patch(facecolor=C_RED    + "33", edgecolor=C_RED,    label="Security & privacy bench. (eICU-CRD Demo)"),
    mpatches.Patch(facecolor=C_ORANGE + "33", edgecolor=C_ORANGE, label="Engagement-policy module"),
    mpatches.Patch(facecolor=C_PURPLE + "33", edgecolor=C_PURPLE, label="Privacy evaluation"),
]
ax.legend(handles=patches, loc="lower left", fontsize=7,
          framealpha=0.85, edgecolor="#CCCCCC", ncol=2,
          bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
