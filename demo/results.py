"""Read-only presentation of the frozen Test experiment and CPU benchmark."""
import numpy as np
import pandas as pd

from demo.data import bundle_index, verify_hashes
from demo.runtime import ROOT, DemoError, read_json


def load_research_results(root=ROOT):
    bundle = bundle_index(root)
    verify_hashes(root, bundle["frozen_files"])
    final = read_json(root / "results/final_test/index.json")
    benchmark = read_json(root / "results/benchmark/index.json")
    lock = read_json(root / "results/protocol_lock.json")
    if not final.get("passed") or not benchmark.get("passed") or final["protocol_sha256"] != lock["protocol_sha256"]:
        raise DemoError("Research summary is not a passed frozen experiment")
    rows, per_seed, by_subject = [], [], []
    for method in ("E0", "E1", "E2", "E3", "E4"):
        keys = [method] if method in ("E0", "E1") else [f"{method}_{seed}" for seed in (42, 123, 999)]
        metrics = [read_json(root / final["methods"][key]["metrics"]) for key in keys]
        macro = [metric["overall"]["macro_subject"] for metric in metrics]
        row = {"method": method, "n_seeds": len(keys), "n_windows": metrics[0]["overall"]["pooled_window"]["n_total"],
               "coverage": min(m["overall"]["pooled_window"]["coverage"] for m in metrics)}
        for name in ("mae_bpm", "rmse_bpm", "error_gt_10_percent"):
            values = [m[name] for m in macro]
            row[name] = float(np.mean(values))
            row[name + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
        for key, metric in zip(keys, metrics):
            per_seed.append({"method": method, "seed": int(key.split("_")[1]) if "_" in key else None,
                             **metric["overall"]["macro_subject"]})
            for subject in metric["by_subject"]:
                by_subject.append({"method": method, "seed": int(key.split("_")[1]) if "_" in key else None, **subject})
    latency = pd.read_csv(root / benchmark["tables"]["model_benchmark.csv"])
    pairs = pd.read_csv(root / benchmark["tables"]["paired_benchmark.csv"])
    if (latency.batch_size != 1).any() or (latency.warmup < 100).any() or (latency.n < 1000).any():
        raise DemoError("CPU benchmark does not satisfy the declared measurement protocol")
    return {"summary": pd.DataFrame(rows), "per_seed": pd.DataFrame(per_seed),
            "by_subject": pd.DataFrame(by_subject), "latency": latency, "pairs": pairs,
            "runtime": benchmark["runtime"], "final_index": "results/final_test/index.json",
            "benchmark_index": "results/benchmark/index.json", "scope": "frozen full Test experiment; not session replay statistics"}
