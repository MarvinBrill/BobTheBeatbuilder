"""Optional within-song motif ablation, with explicit reference-timing diagnostics."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "data/benchmarks/architecture/training"
os.environ["BOB_DATA"] = str(DIRECTORY / "runtime")
sys.path.insert(0, str(DIRECTORY / "source"))
from app.ml.motif_memory import analysis_spans, link_spans, MotifMemory
from app.ml.rollout_evaluation import rollout_settings, roundtrip_report
from app.ml.audio import FRAME_MS
from architecture_experiment_data import prepare_manifest


class Context:
    def check(self):
        pass

    def report(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--arm", default="critic")
    parser.add_argument("--seeds", default="42,73")
    args = parser.parse_args()
    _, validation, corpus_hash = prepare_manifest(ROOT / "data/benchmarks/architecture/c-training-records.json", Context())
    analysis = []
    for item in validation:
        started = time.perf_counter()
        spans = analysis_spans(item["mel"], item["timing"])
        links = link_spans(item["mel"], spans)
        analysis.append(dict(map_id=item["map_id"], spans=len(spans), links=[asdict(link) for link in links],
                             seconds=time.perf_counter() - started))
    report = dict(schema="motif-ablation-v1", corpus_sha256=corpus_hash, analysis=analysis, results=[],
                  note="Ordered within-song spectral hypotheses. Reference timing is an explicitly optimistic manual-timing diagnostic. No human or musical-phrase labels; no final test.")
    output = ROOT / "data/benchmarks/architecture/motif-comparison.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Reference-timing motif links", {row["map_id"]: len(row["links"]) for row in analysis}, flush=True)
    if args.analysis_only:
        return

    import torch
    from safetensors.torch import load_file
    from app.ml.phrase_model import create_model
    from app.ml.sequence_generation import generate_objects
    from app.ml.generation_trace import GenerationTrace

    torch.set_num_threads(4)
    # Fixed maximum of two songs, prioritising a probe where the optional mechanism
    # can actually operate. This is a mechanism test, not a representative quality sample.
    selected = sorted(validation, key=lambda item: (-next(len(row["links"]) for row in analysis if row["map_id"] == item["map_id"]), item["map_id"]))[:2]
    for seed in map(int, args.seeds.split(",")):
        directory = DIRECTORY / f"{args.arm}-{seed}"
        metadata = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        model = create_model(metadata["architecture"], metadata["model_config"]).to("cuda").eval()
        model.load_state_dict(load_file(str(directory / "model.safetensors")))
        for item in selected:
            for timing_mode in ("estimated", "reference_manual"):
                settings = rollout_settings(item["condition"], seed)
                if timing_mode == "reference_manual":
                    settings["timing_points"] = [asdict(point) for point in item["timing"]]
                for enabled in (False, True):
                    trace = GenerationTrace()
                    started = time.perf_counter()
                    objects, timing, quality = generate_objects(
                        model, item["mel"], item["audio_path"], {**settings, "motifs": enabled},
                        torch.tensor(item["condition"][None], device="cuda"), torch.device("cuda"), Context(),
                        max_objects=1024, trace=trace,
                    )
                    memory = MotifMemory(link_spans(item["mel"], analysis_spans(item["mel"], timing)))
                    relative_costs = [memory.cost(obj, objects[:index]) for index, obj in enumerate(objects)]
                    record = dict(map_id=item["map_id"], seed=seed, timing=timing_mode, motifs=enabled,
                                  seconds=time.perf_counter() - started, **quality,
                                  mean_relative_motif_cost=float(np.mean(relative_costs)) if relative_costs else 0,
                                  affected_candidate_scores=sum(event["stage"] == "candidate_score" and event.get("weighted_motif", 0) < 0 for event in trace.events),
                                  **roundtrip_report(objects, timing, settings))
                    report["results"].append(record)
                    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                    print(seed, item["map_id"], timing_mode, enabled, "links", quality["motif_links"], "heads", len(objects), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
