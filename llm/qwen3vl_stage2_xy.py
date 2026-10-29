#!/usr/bin/env python3
"""Classify worn gown, mask, and eyewear using saved Stage 1 predictions."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.qwen3vl_policy_nonadherence import crop_view, extract_json, run_one  # noqa: E402
from llm.qwen3vl_stage1_xy import save_json  # noqa: E402


CLASS_IDS = {
    "gown": ("Gown", ("GC", "GI")),
    "mask": ("Mask", ("NC", "RC", "MI", "PR")),
    "eyewear": ("Eyewear", ("PR", "GG", "FC", "FI", "SG")),
}
SYSTEM_PROMPT = "Classify the target clinician's visible PPE across three camera views. Return valid JSON only."
PROMPT_VERSION = "present_classification_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="outputs/llm/qwen3vl_stage1_xy_half_seed7.json")
    parser.add_argument("--meta", default="dataset/five_simulation/meta.json")
    parser.add_argument("--output", default="outputs/llm/qwen3vl_stage2_xy_half_seed7.json")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def descriptions(meta: dict) -> dict[str, dict[str, str]]:
    source = meta["attributeDescriptions"]
    return {
        key: {label: source[attribute][label] for label in labels}
        for key, (attribute, labels) in CLASS_IDS.items()
    }


def prompt_for(requested: list[str], class_descriptions: dict[str, dict[str, str]]) -> str:
    candidates = {key: class_descriptions[key] for key in requested}
    return (
        "Look at all three views of the same clinician. Classify only these items that an earlier "
        "step marked as worn. Use the visual descriptions below; no policy judgment is needed.\n"
        f"Class descriptions: {json.dumps(candidates, ensure_ascii=False)}\n"
        "For each requested item, return exactly one listed class ID. Return 'none' if the item "
        "is not actually worn, or 'unclear' if the views do not allow a reliable choice. "
        "Return one JSON object with only the requested item names as keys and class IDs as values."
    )


def build_items(stage1: dict, limit: int | None) -> list[dict]:
    source_items = stage1["items"][:limit] if limit is not None else stage1["items"]
    items = []
    for source in source_items:
        if "prediction" not in source:
            raise ValueError(f"Stage 1 prediction missing: {source['sample_id']}")
        requested = [key for key in CLASS_IDS if source["prediction"][key] == "present"]
        ground_truth = {
            key: sorted({view["view_ground_truth"][attribute] for view in source["views"]})
            for key, (attribute, _) in CLASS_IDS.items()
        }
        items.append({
            "sample_index": source["sample_index"],
            "sample_id": source["sample_id"],
            "views": source["views"],
            "stage1_prediction": source["prediction"],
            "ground_truth": ground_truth,
            "requested_attributes": requested,
        })
    return items


def main() -> None:
    args = parse_args()
    stage1 = json.loads(Path(args.stage1).read_text())
    class_descriptions = descriptions(json.loads(Path(args.meta).read_text()))
    expected_items = build_items(stage1, args.limit)
    output = Path(args.output)
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists; use --resume: {output}")
        results = json.loads(output.read_text())
        if (
            results["model"] != args.model
            or results["prompt_version"] != PROMPT_VERSION
            or [item["sample_id"] for item in results["items"]]
            != [item["sample_id"] for item in expected_items]
        ):
            raise ValueError("Saved run does not match model, prompt, or samples")
    else:
        results = {
            "stage1_source": args.stage1,
            "meta_source": args.meta,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "class_descriptions": class_descriptions,
            "items": expected_items,
        }
        save_json(output, results)
    total = len(results["items"])
    print(f"selected {total} Stage 1 samples", flush=True)
    if args.sample_only:
        return

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    for item in results["items"]:
        if "prediction" in item:
            continue
        requested = item["requested_attributes"]
        if not requested:
            item["prediction"] = {}
            save_json(output, results)
            continue
        repair = "raw_output" in item and "parse_error" in item
        previous = {}
        if repair:
            try:
                parsed_previous = extract_json(item["raw_output"])
                previous = {
                    key: value
                    for key in requested
                    if isinstance(value := parsed_previous.get(key), str)
                    and value.strip().upper() in class_descriptions[key]
                }
                previous = {key: value.strip().upper() for key, value in previous.items()}
            except (TypeError, ValueError):
                pass
            requested = [key for key in requested if key not in previous]
            item["repair_requested_attributes"] = requested
        try:
            with tempfile.TemporaryDirectory(prefix="qwen3vl_stage2_") as temporary:
                paths = [
                    crop_view(
                        item,
                        view,
                        Path(view["image_path"]).parent,
                        Path(temporary),
                        item["sample_index"],
                    )
                    for view in item["views"]
                ]
                raw = run_one(
                    model,
                    processor,
                    paths,
                    [],
                    prompt_for(requested, class_descriptions),
                    args.max_new_tokens,
                    0.0,
                    False,
                    system_prompt=SYSTEM_PROMPT,
                )
        except Exception as exc:
            item["inference_error"] = f"{type(exc).__name__}: {exc}"
            save_json(output, results)
            raise
        item["repair_raw_output" if repair else "raw_output"] = raw
        try:
            parsed = extract_json(raw)
            prediction = previous.copy()
            for key in requested:
                value = parsed.get(key)
                if not isinstance(value, str):
                    raise ValueError(f"Missing class for {key}")
                value = value.strip()
                value = value.lower() if value.lower() in {"none", "unclear"} else value.upper()
                if value not in class_descriptions[key] and value not in {"none", "unclear"}:
                    raise ValueError(f"Invalid class for {key}: {value}")
                prediction[key] = value
            item["prediction"] = prediction
            item.pop("parse_error", None)
        except (TypeError, ValueError) as exc:
            item["repair_parse_error" if repair else "parse_error"] = f"{type(exc).__name__}: {exc}"
        save_json(output, results)
        print(f"completed {item['sample_index'] + 1}/{total}: {item['sample_id']}", flush=True)


if __name__ == "__main__":
    main()
