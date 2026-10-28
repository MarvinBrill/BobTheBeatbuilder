"""Package generated A/B maps without revealing conditions or inventing responses.

Input schema: blind-playtest-input-v1, pairs with song_id, seed, section, style,
left/right objects containing map_path, audio_path, condition. The private key
is stored outside the participant directory. Original generated maps are untouched.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = (
    "timing_and_rhythm",
    "musical_interpretation",
    "motifs_and_transitions",
    "readability",
    "fatigue",
    "enjoyment",
)


def anonymous_map(source, pair_id, label):
    lines, section = [], ""
    for line in Path(source).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("["):
            section = line
            lines.append(line)
            if section == "[Metadata]":
                lines.extend(
                    (
                        f"Title:Playtest {pair_id}",
                        "Artist:Anonymous",
                        "Creator:Anonymous",
                        f"Version:{label}",
                        "BeatmapID:0",
                        "BeatmapSetID:-1",
                    )
                )
        elif section == "[Metadata]":
            continue
        elif section == "[General]" and line.startswith("AudioFilename:"):
            lines.append("AudioFilename: audio.ogg")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def plan_pairs(pairs, seed):
    rng = random.Random(seed)
    ordered = sorted(
        pairs,
        key=lambda item: (
            item["song_id"],
            item["seed"],
            json.dumps(item["section"], sort_keys=True),
        ),
    )
    planned, keys = [], []
    for index, pair in enumerate(ordered):
        identity = json.dumps(
            [pair["song_id"], pair["seed"], pair["section"], index], sort_keys=True
        )
        pair_id = hashlib.sha256(identity.encode()).hexdigest()[:12]
        sides = ["left", "right"]
        rng.shuffle(sides)
        labels = dict(zip(("A", "B"), sides))
        order = ["A", "B"]
        rng.shuffle(order)
        planned.append(dict(pair_id=pair_id, section=pair["section"], play_order=order))
        keys.append(
            dict(
                pair_id=pair_id,
                song_id=pair["song_id"],
                seed=pair["seed"],
                style=pair.get("style"),
                assignments={label: pair[side] for label, side in labels.items()},
            )
        )
    return planned, keys


def questionnaire(planned):
    return {
        "schema": "blind-playtest-responses-v1",
        "instructions": "Play both versions in the assigned order with equal settings. Preference and confidence are independent. Record concrete audio timestamps; leave unanswered fields null. For fatigue, prefer the less tiring version only when that supports the music and target difficulty.",
        "allowed_preferences": ["A", "B", "equal", "unable_to_judge"],
        "confidence_scale": {"1": "very uncertain", "5": "very certain"},
        "participant_id": None,
        "responses": [
            dict(
                **pair,
                overall_preference=None,
                confidence=None,
                dimensions={name: {"preference": None, "confidence": None} for name in DIMENSIONS},
                observations=[],
                observation_schema={
                    "audio_timestamp_ms": "number",
                    "end_timestamp_ms": "optional number",
                    "dimension": "one of dimensions",
                    "version": "A/B/both",
                    "comment": "free text",
                },
            )
            for pair in planned
        ],
    }


def package(manifest, output, key_path, seed):
    if manifest.get("schema") != "blind-playtest-input-v1":
        raise ValueError("Unsupported playtest input schema; expected blind-playtest-input-v1")
    if key_path.resolve().is_relative_to(output.resolve()):
        raise ValueError("The private condition key must be outside the participant directory")
    if output.exists() or key_path.exists():
        raise ValueError("Use fresh output/key paths to preserve an existing blinded study")
    planned, keys = plan_pairs(manifest["pairs"], seed)
    # Validate all sources before creating the participant package.
    for key in keys:
        for source in key["assignments"].values():
            if not Path(source["map_path"]).is_file() or not Path(source["audio_path"]).is_file():
                raise ValueError("Missing generated map or source audio")
        if (
            Path(key["assignments"]["A"]["audio_path"]).resolve()
            != Path(key["assignments"]["B"]["audio_path"]).resolve()
        ):
            raise ValueError("A/B maps must use the same source audio file")
    output.mkdir(parents=True)
    os.environ["BOB_DATA"] = str(output.parent / ".playtest-runtime")
    sys.path.insert(0, str(ROOT / "backend"))
    from app.ml.audio import export_audio
    from app.beatmaps import parse

    for pair, key in zip(planned, keys):
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            directory = Path(temporary)
            export_audio(Path(key["assignments"]["A"]["audio_path"]), directory / "audio.ogg")
            for label, source in key["assignments"].items():
                target = directory / f"{label}.osu"
                target.write_text(
                    anonymous_map(source["map_path"], pair["pair_id"], label), encoding="utf-8"
                )
                before, after = parse(Path(source["map_path"])), parse(target)
                if before.objects != after.objects or before.timing != after.timing:
                    raise ValueError("Blinding changed gameplay content")
            archive_path = output / f"{pair['pair_id']}.osz"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for name in ("audio.ogg", "A.osu", "B.osu"):
                    archive.write(directory / name, name)
            pair["archive"] = archive_path.name
    public = {
        "schema": "blind-playtest-package-v1",
        "pairs": planned,
        "human_results": "not_collected",
    }
    (output / "manifest.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    (output / "questionnaire.json").write_text(
        json.dumps(questionnaire(planned), indent=2), encoding="utf-8"
    )
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(
        json.dumps(
            {"schema": "blind-playtest-key-v1", "randomization_seed": seed, "pairs": keys}, indent=2
        ),
        encoding="utf-8",
    )
    return public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    public = package(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        args.output,
        args.private_key,
        args.seed,
    )
    print(f"Prepared {len(public['pairs'])} blinded pairs. Human responses remain empty.")


if __name__ == "__main__":
    main()
