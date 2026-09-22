"""Master runner — generate all six TRUST-CT manuscript figures (figs 1-6).

Figs 1-5: trust-ct/_pkg_work/manuscript/figure_scripts/
Fig 6:    Fedlearn/manuscript/figure_scripts/fig6_e6_fairness.py
"""
import os
import importlib.util
import traceback

HERE     = os.path.dirname(os.path.abspath(__file__))
FIG6_PY  = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "..", "manuscript", "figure_scripts", "fig6_e6_fairness.py")
)

SCRIPTS = [
    os.path.join(HERE, "fig1_workflow.py"),
    os.path.join(HERE, "fig2_ni_forest.py"),
    os.path.join(HERE, "fig3_policy_replay.py"),
    os.path.join(HERE, "fig4_attacks.py"),
    os.path.join(HERE, "fig5_frontier.py"),
    FIG6_PY,
]

passed, failed = [], []
for path in SCRIPTS:
    name = os.path.basename(path)
    print(f"\n{'='*55}\nRunning {name} ...\n{'='*55}")
    try:
        spec = importlib.util.spec_from_file_location("_fig_mod", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        passed.append(name)
    except Exception:
        print(f"ERROR in {name}:")
        traceback.print_exc()
        failed.append(name)

print(f"\n{'='*55}")
print(f"Done. {len(passed)}/{len(SCRIPTS)} figures generated successfully.")
if passed:
    print("Passed:", ", ".join(passed))
if failed:
    print("FAILED:", ", ".join(failed))
