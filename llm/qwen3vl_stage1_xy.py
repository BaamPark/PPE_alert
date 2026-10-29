#!/usr/bin/env python3
"""Run the first-stage PPE questions on random three-view samples from xy.json."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.qwen3vl_policy_nonadherence import crop_view, extract_json, find_image, run_one  # noqa: E402
from llm.qwen3vl_visual_retrieval_eval import (  # noqa: E402
    PRESENCE_PROMPT,
    PRESENCE_SYSTEM_PROMPT,
)
from tools.build_policy_eval_samples import group_rows  # noqa: E402

OUTPUT_KEYS = ("gown", "mask", "eyewear", "left_glove", "right_glove")
VALID_STATES = {"present", "absent", "unclear"}
PROMPT_VERSION = "simple_presence_v4"
GROUND_TRUTH_CLASSES = {
    "gown": ("Gown", {"GC", "GI"}, {"GA"}),
    "mask": ("Mask", {"NC", "RC", "MI", "PR"}, {"MA"}),
    "eyewear": ("Eyewear", {"PR", "GG", "FC", "FI", "SG"}, {"EA"}),
    "left_glove": ("Glove-L", {"HC"}, {"HA"}),
    "right_glove": ("Glove-R", {"HC"}, {"HA"}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/five_simulation/xy.json")
    parser.add_argument("--output", default="outputs/llm/qwen3vl_stage1_xy_100_seed7.json")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def select_samples(rows: list[dict], count: int, seed: int) -> tuple[list[dict], int]:
    groups = group_rows(rows)
    candidates = [
        (key, views)
        for key, views in sorted(groups.items())
        if len(views) == 3 and {view["view"] for view in views} == {0, 1, 2}
    ]
    if count < 1 or count > len(candidates):
        raise ValueError(f"Requested {count} samples, but only {len(candidates)} complete triplets exist")
    chosen = random.Random(seed).sample(candidates, count)
    samples = [
        {
            "sample_id": f"{simulation}_frame{frame_index:03d}_id{clinician_id}",
            "simulation": simulation,
            "frame_index": frame_index,
            "clinician_id": clinician_id,
            "views": [
                {
                    "view": view["view"],
                    "file": view["file"],
                    "bbox": view["bbox"],
                    "view_ground_truth": view["attributes"],
                }
                for view in views
            ],
        }
        for (simulation, frame_index, clinician_id), views in chosen
    ]
    return samples, len(candidates)


def sample_ground_truth(sample: dict) -> dict[str, str]:
    ground_truth = {}
    for key, (attribute, present, absent) in GROUND_TRUTH_CLASSES.items():
        labels = {view["view_ground_truth"][attribute] for view in sample["views"]}
        ground_truth[key] = (
            "present" if labels & present else "absent" if labels & absent else "unclear"
        )
    return ground_truth


def main() -> None:
    args = parse_args()
    samples, candidate_count = select_samples(json.loads(Path(args.input).read_text()), args.num_samples, args.seed)
    output = Path(args.output)
    sample_ids = [sample["sample_id"] for sample in samples]
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists; use --resume: {output}")
        results = json.loads(output.read_text())
        if (
            [item["sample_id"] for item in results["items"]] != sample_ids
            or results["presence_prompt_version"] != PROMPT_VERSION
            or results["model"] != args.model
        ):
            raise ValueError("Saved run does not match the requested samples, prompt, or model")
    else:
        image_root = Path(args.image_root) if args.image_root else None
        for sample in samples:
            for view in sample["views"]:
                view["image_path"] = str(find_image(image_root, view["file"]).resolve())
        results = {
            "input": args.input,
            "model": args.model,
            "seed": args.seed,
            "num_samples": len(samples),
            "complete_triplet_candidates": candidate_count,
            "samples_per_simulation": dict(Counter(sample["simulation"] for sample in samples)),
            "presence_prompt_version": PROMPT_VERSION,
            "system_prompt": PRESENCE_SYSTEM_PROMPT,
            "user_prompt": PRESENCE_PROMPT,
            "items": [
                {"sample_index": index, **sample, "ground_truth": sample_ground_truth(sample)}
                for index, sample in enumerate(samples)
            ],
        }

    save_json(output, results)
    print(f"selected {len(samples)} of {candidate_count} complete triplets: {results['samples_per_simulation']}", flush=True)
    if args.sample_only:
        return

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    for item in results["items"]:
        if "raw_output" in item:
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="qwen3vl_stage1_") as temporary:
                crop_paths = [
                    crop_view(item, view, Path(view["image_path"]).parent, Path(temporary), item["sample_index"])
                    for view in item["views"]
                ]
                raw = run_one(
                    model,
                    processor,
                    crop_paths,
                    [],
                    PRESENCE_PROMPT,
                    args.max_new_tokens,
                    0.0,
                    False,
                    system_prompt=PRESENCE_SYSTEM_PROMPT,
                )
        except Exception as exc:
            item["inference_error"] = f"{type(exc).__name__}: {exc}"
            save_json(output, results)
            raise
        item["raw_output"] = raw
        try:
            parsed = extract_json(raw)
            if any(parsed.get(key) not in VALID_STATES for key in OUTPUT_KEYS):
                raise ValueError("Missing or invalid presence label")
            item["prediction"] = {key: parsed[key] for key in OUTPUT_KEYS}
        except (TypeError, ValueError) as exc:
            item["parse_error"] = f"{type(exc).__name__}: {exc}"
        save_json(output, results)
        print(f"completed {item['sample_index'] + 1}/{len(samples)}: {item['sample_id']}", flush=True)


if __name__ == "__main__":
    main()
