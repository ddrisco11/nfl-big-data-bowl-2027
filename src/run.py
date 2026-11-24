"""Reproduce native-rate route models and the integrated unpredictability analysis."""

import argparse, os, pathlib, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument(
    "--data", required=True, help="Directory containing the nine competition CSVs"
)
p.add_argument("--work", default=".work")
p.add_argument("--output", default=".work/results")
a = p.parse_args()
os.environ["NFL_DATA_DIR"] = str(pathlib.Path(a.data).resolve())
root = pathlib.Path(a.work).resolve()
for key, value in [
    ("NFL_BASE_WORK", root / "base"),
    ("NFL_MODEL_WORK", root / "models"),
    ("NFL_OUTPUT_DIR", pathlib.Path(a.output).resolve()),
]:
    os.environ[key] = str(value)
    value.mkdir(parents=True, exist_ok=True)
for name in [
    "prepare_base",
    "prepare",
    "trees",
    "sequences",
    "tcn",
    "evaluate",
    "integrate",
    "combine_features",
    "associate",
    "stability",
]:
    subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).with_name(name + ".py"))],
        check=True,
    )
