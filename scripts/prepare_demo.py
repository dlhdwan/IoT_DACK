"""Prepare every locked Test window for the lecturer's presentation UI."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.data import export_test_bundle
from demo.paths import artifact_root

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Locked notebook workspace; defaults to PPG_RUN_DIR or configs/primary_run.json")
    args = parser.parse_args()
    index = export_test_bundle(artifact_root(args.run_dir))
    print(f"PASS: {index['n_windows']} Test windows, S13–S15, all eight activities; no inference/training")
    print(f"Feature bundle: {index['feature_bytes']:,} bytes; {index['features']}")
