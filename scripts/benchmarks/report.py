"""Summarize saved measurements without rerunning training or changing acceptance limits."""

import json
import statistics
from corpus import DIRECTORY


def read(name):
    return json.loads((DIRECTORY / f"{name}.json").read_text(encoding="utf-8"))


def quality_report():
    protocol, results = read("quality-protocol"), read("quality")
    modes = ("baseline", "fp32", "bf16", "graph")
    expected = {(seed, mode) for seed in protocol["seeds"] for mode in modes}
    assert {(row["seed"], row["mode"]) for row in results} == expected, "Incomplete quality experiment"
    assert all(row["updates"] == protocol["updates"] for row in results)
    metrics = ("loss", "onset_f1", "motion_mae", "object_type_accuracy")
    averages = {
        mode: {
            key: statistics.mean(row["history"][-1][key] for row in results if row["mode"] == mode)
            for key in metrics
        }
        for mode in modes
    }
    reference = averages["baseline"]
    terms = {
        mode: {
            key: statistics.mean(
                row["history"][-1]["loss_terms"][key] for row in results if row["mode"] == mode
            )
            for key in results[0]["history"][-1]["loss_terms"]
        }
        for mode in modes
    }
    rollout = {
        mode: {
            key: statistics.mean(row["free_generation"][key] for row in results if row["mode"] == mode)
            for key in (
                "reference_onset_f1",
                "temporal_overlaps",
                "heavily_repaired_fraction",
                "mean_geometry_repair",
                "jump_p95_pixels",
                "aim_speed_p95",
            )
        }
        for mode in modes
    }
    gates = {}
    for mode in modes[1:]:
        measured = averages[mode]
        gates[mode] = dict(
            validation_loss=measured["loss"] <= reference["loss"] + max(0.03, 0.02 * abs(reference["loss"])),
            onset_f1=measured["onset_f1"] >= reference["onset_f1"] - 0.02,
            motion_mae=measured["motion_mae"] <= reference["motion_mae"] * 1.05,
        )
    return dict(
        protocol=protocol,
        averages=averages,
        loss_terms=terms,
        free_generation=rollout,
        regression_gates=gates,
        passed=all(all(values.values()) for values in gates.values()),
    )


def throughput_report():
    results = read("throughput-sustained")
    grouped = {}
    for row in results:
        key = f'{row["mode"]}_batch{row["batch_size"]}_acc{row["accumulation"]}'
        grouped.setdefault(key, []).append(row)
    return {
        key: dict(
            repetitions=len(rows),
            median_examples_per_second=statistics.median(row["examples_per_second"] for row in rows),
            median_gpu_utilization=statistics.median(row["gpu_utilization_mean"] for row in rows),
            peak_allocated_mib=max(row["peak_allocated_mib"] for row in rows),
            peak_reserved_mib=max(row["peak_reserved_mib"] for row in rows),
            warmup_peak_allocated_mib=max(row["warmup_peak_allocated_mib"] for row in rows),
            warmup_peak_reserved_mib=max(row["warmup_peak_reserved_mib"] for row in rows),
            device_memory_peak_mib=max(row["device_memory_peak_mib"] for row in rows),
        )
        for key, rows in grouped.items()
    }


if __name__ == "__main__":
    result = dict(quality=quality_report(), throughput=throughput_report(), parity=read("parity"))
    (DIRECTORY / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))
