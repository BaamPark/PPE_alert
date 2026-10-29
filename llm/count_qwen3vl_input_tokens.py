#!/usr/bin/env python3
"""Count Qwen3-VL processor input tokens for the policy-eval prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoProcessor

from qwen3vl_policy_nonadherence import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    collect_reference_images,
    crop_view,
    load_json,
    sample_items,
    select_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--samples-json", default="dataset/five_simulation/y1_policy_eval_samples.json")
    parser.add_argument("--ppe-description", default="dataset/ppe_description.json")
    parser.add_argument("--policy-set", default="dataset/policy_set.json")
    parser.add_argument("--reference-dir", default="dataset/reference")
    parser.add_argument("--use-reference-images", action="store_true")
    parser.add_argument("--requirement-level", default="R4")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--crop-dir", default="outputs/llm/qwen3vl_token_count_crops")
    parser.add_argument("--simulation", default="sim1")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_content(
    crop_paths: list[Path],
    reference_images: list[tuple[str, Path]],
    prompt: str,
) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    if reference_images:
        content.append(
            {
                "type": "text",
                "text": "Reference images below show visual examples of PPE categories.",
            }
        )
        for label, path in reference_images:
            content.append({"type": "image", "image": str(path)})
            content.append({"type": "text", "text": f"Reference PPE category: {label}."})
        content.append(
            {
                "type": "text",
                "text": "Query images below are cropped views of the same clinician to evaluate.",
            }
        )
    for idx, path in enumerate(crop_paths):
        content.append({"type": "image", "image": str(path)})
        content.append({"type": "text", "text": f"Image {idx}: view crop from the same clinician."})
    content.append({"type": "text", "text": prompt})
    return content


def tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return list(shape)


def main() -> None:
    args = parse_args()
    sample_data = load_json(args.samples_json, {})
    ppe_descriptions = load_json(args.ppe_description, {})
    policy = select_policy(load_json(args.policy_set, []), args.requirement_level)
    samples = sample_items(sample_data["samples"], args.simulation, args.sample_index + 1, args.seed)
    sample = samples[args.sample_index]
    image_root = Path(args.image_root) if args.image_root else None
    crop_dir = Path(args.crop_dir)
    crop_paths = [
        crop_view(sample, view_entry, image_root, crop_dir, args.sample_index)
        for view_entry in sample["views"]
    ]
    reference_images: list[tuple[str, Path]] = []
    missing_reference_labels: list[str] = []
    if args.use_reference_images:
        reference_images, missing_reference_labels = collect_reference_images(
            Path(args.reference_dir),
            ppe_descriptions,
        )
    prompt = USER_PROMPT_TEMPLATE.format(
        ppe_descriptions=json.dumps(ppe_descriptions, indent=2),
        requirement_description=policy["requirement_description"],
        violation_visual_cues=json.dumps(policy["violation_visual_cues"], indent=2),
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": build_content(crop_paths, reference_images, prompt)},
    ]
    processor = AutoProcessor.from_pretrained(args.model)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    image_grid_thw = inputs.get("image_grid_thw")
    result = {
        "model": args.model,
        "requirement_level": args.requirement_level,
        "sample_id": sample["sample_id"],
        "num_reference_images": len(reference_images),
        "num_query_images": len(crop_paths),
        "missing_reference_labels": missing_reference_labels,
        "input_ids_shape": tensor_shape(input_ids),
        "input_token_count": int(input_ids.shape[-1]),
        "pixel_values_shape": tensor_shape(inputs.get("pixel_values")),
        "image_grid_thw": image_grid_thw.tolist() if image_grid_thw is not None else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
