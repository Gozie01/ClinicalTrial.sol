"""Generate PNG versions of all five figures (300 dpi)."""
import os
import sys
import importlib.util
import traceback
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Patch plt.savefig so every script saves .png instead of .pdf
_orig_savefig = plt.savefig
def _png_savefig(path, **kwargs):
    png_path = str(path).replace(".pdf", ".png")
    kwargs.setdefault("dpi", 300)
    kwargs["bbox_inches"] = kwargs.get("bbox_inches", "tight")
    _orig_savefig(png_path, **kwargs)
    print(f"  -> PNG: {png_path}")
plt.savefig = _png_savefig

SCRIPTS = [
    "fig1_workflow.py",
    "fig2_ni_forest.py",
    "fig3_policy_replay.py",
    "fig4_attacks.py",
    "fig5_frontier.py",
]

passed, failed = [], []
for script in SCRIPTS:
    path = os.path.join(HERE, script)
    print(f"\n{'='*50}\n{script}\n{'='*50}")
    try:
        spec = importlib.util.spec_from_file_location("_fig_mod", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        passed.append(script)
    except Exception:
        traceback.print_exc()
        failed.append(script)

print(f"\nDone. {len(passed)}/{len(SCRIPTS)} figures generated.")
figures_dir = os.path.abspath(os.path.join(HERE, "..", "figures"))
pngs = sorted(f for f in os.listdir(figures_dir) if f.endswith(".png"))
print(f"PNGs in {figures_dir}:")
for p in pngs:
    print(f"  {p}")
