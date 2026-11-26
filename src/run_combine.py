"""Run Combine-to-separation comparisons after the main preparation pipeline."""

import argparse, os, pathlib, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--work", default=".work")
p.add_argument("--output", default=".work/combine-upgrades")
a = p.parse_args()
w = pathlib.Path(a.work).resolve()
os.environ["NFL_DATA_DIR"] = str(pathlib.Path(a.data).resolve())
os.environ["NFL_BASE_WORK"] = str(w / "base")
os.environ["NFL_OUTPUT_DIR"] = str(w / "results")
os.environ["NFL_COMBINE_OUTPUT"] = str(pathlib.Path(a.output).resolve())
pathlib.Path(a.output).mkdir(parents=True, exist_ok=True)
for n in ["prepare", "timing", "analyze", "summarize", "report"]:
    subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).with_name("combine_" + n + ".py"))],
        check=True,
    )
