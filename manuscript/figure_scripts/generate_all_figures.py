"""Master runner — generate all five TRUST-CT manuscript figures."""
import os
import sys
import importlib.util
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
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
    print(f"\n{'='*55}\nRunning {script} ...\n{'='*55}")
    try:
        spec = importlib.util.spec_from_file_location("_fig_mod", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        passed.append(script)
    except Exception:
        print(f"ERROR in {script}:")
        traceback.print_exc()
        failed.append(script)

print(f"\n{'='*55}")
print(f"Done. {len(passed)}/{len(SCRIPTS)} figures generated successfully.")
if passed:
    print("Passed:", ", ".join(passed))
if failed:
    print("Failed:", ", ".join(failed))

figures_dir = os.path.join(HERE, "..", "figures")
print(f"\nOutput directory: {os.path.abspath(figures_dir)}")
if os.path.isdir(figures_dir):
    pdfs = [f for f in os.listdir(figures_dir) if f.endswith(".pdf")]
    print(f"PDFs present: {sorted(pdfs)}")
