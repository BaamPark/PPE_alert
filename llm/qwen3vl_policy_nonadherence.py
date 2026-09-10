#!/usr/bin/env python3
"""Evaluate Qwen3-VL policy-conditioned PPE nonadherence reasoning."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image


FRAME_RE = re.compile(r"^(sim\d+)_view_(\d+)_frame(\d+)\.")

SYSTEM_PROMPT = """
You are a PPE policy adherence evaluator.
You receive cropped images of the same clinician from multiple camera views.
Use the PPE descriptions, requirement description, and violation visual cues to decide whether each violation alert should be raised.
Return only valid JSON. Do not include markdown. Do not explain outside the JSON.
""".strip()

USER_PROMPT_TEMPLATE = """
PPE descriptions:
{ppe_descriptions}

Requirement description:
{requirement_description}

Violation visual cues:
{violation_visual_cues}

Task:
Evaluate the cropped images of the same clinician across views.
Return this exact JSON schema:
{{
  "alerts": {{
    "gown": {{"violation": true or false, "visual_evidence": "short visual evidence"}},
    "mask": {{"violation": true or false, "visual_evidence": "short visual evidence"}},
    "eyewear": {{"violation": true or false, "visual_evidence": "short visual evidence"}},
    "left_glove": {{"violation": true or false, "visual_evidence": "short visual evidence"}},
    "right_glove": {{"violation": true or false, "visual_evidence": "short visual evidence"}}
  }},
  "overall_violation": true or false
}}
""".strip()

OUTPUT_KEY_TO_ALERT_KEY = {
    "gown_nonadherence": "gown",
    "mask_nonadherence": "mask",
    "eyewear_nonadherence": "eyewear",
    "left_glove_nonadherence": "left_glove",
    "right_glove_nonadherence": "right_glove",
}
OUTPUT_KEYS = list(OUTPUT_KEY_TO_ALERT_KEY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--samples-json", default="dataset/five_simulation/y1_policy_eval_samples.json")
    parser.add_argument("--ppe-description", default="dataset/ppe_description.json")
    parser.add_argument("--policy-set", default="dataset/policy_set.json")
    parser.add_argument("--requirement-level", default="R4")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", default="outputs/llm/qwen3vl_policy_nonadherence_10.json")
    parser.add_argument("--crop-dir", default="outputs/llm/qwen3vl_policy_nonadherence_crops")
    parser.add_argument("--simulation", default="sim1")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--sample-only", action="store_true", help="Create samples/crops without model inference.")
    return parser.parse_args()


def load_json(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")


def default_image_roots() -> list[Path]:
    return [
        Path("dataset/five_simulation/images"),
        Path("/home/beomseok/Datasets/five_simulation/images"),
        Path("/home/beomseok/Datasets/five_simulation"),
    ]


def find_image(image_root: Path | None, file_name: str) -> Path:
    roots = [image_root] if image_root is not None else default_image_roots()
    roots = [root for root in roots if root is not None]
    for root in roots:
        direct = root / file_name
        if direct.exists():
            return direct
        matches = list(root.rglob(file_name)) if root.exists() else []
        if matches:
            return matches[0]
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"Could not find {file_name}. Searched: {searched}")


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        return json.loads(text[start : end + 1])


def select_policy(policy_set: list[dict[str, Any]], requirement_level: str) -> dict[str, Any]:
    for policy in policy_set:
        if policy.get("requirement_level") == requirement_level:
            return policy
    raise KeyError(f"Requirement level not found: {requirement_level}")


def sample_items(samples: list[dict[str, Any]], simulation: str, num_samples: int, seed: int) -> list[dict[str, Any]]:
    candidates = [sample for sample in samples if sample["simulation"] == simulation]
    candidates.sort(key=lambda x: x["sample_id"])
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:num_samples]


def crop_view(
    sample: dict[str, Any],
    view_entry: dict[str, Any],
    image_root: Path | None,
    crop_dir: Path,
    sample_index: int,
) -> Path:
    image_path = find_image(image_root, view_entry["file"])
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        x0, y0, x1, y1 = [int(round(v)) for v in view_entry["bbox"]]
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        crop = image.crop((x0, y0, x1, y1))
        out_dir = crop_dir / f"sample_{sample_index:03d}_{sample['sample_id']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"view{view_entry['view']}_{Path(view_entry['file']).stem}.jpg"
        crop.save(out_path)
        return out_path


def prediction_alert(parsed: dict[str, Any], key: str) -> bool | None:
    alert_key = OUTPUT_KEY_TO_ALERT_KEY[key]
    alerts = parsed.get("alerts")
    if isinstance(alerts, dict):
        value = alerts.get(alert_key)
        if isinstance(value, dict) and isinstance(value.get("violation"), bool):
            return value["violation"]
    return None


def evaluate_prediction(parsed: dict[str, Any] | None, gt: dict[str, Any]) -> dict[str, Any]:
    per_key = {}
    correct = 0
    total = 0
    for key in OUTPUT_KEYS:
        gt_alert = gt[key]["alert"]
        pred_alert = prediction_alert(parsed or {}, key)
        is_correct = pred_alert == gt_alert if pred_alert is not None else None
        if is_correct is not None:
            total += 1
            correct += int(is_correct)
        per_key[key] = {
            "prediction": pred_alert,
            "ground_truth": gt_alert,
            "correct": is_correct,
        }
    pred_overall = (parsed or {}).get("overall_violation")
    overall_gt = gt["overall_nonadherence"]["alert"]
    per_key["overall_nonadherence"] = {
        "prediction": pred_overall if isinstance(pred_overall, bool) else None,
        "ground_truth": overall_gt,
        "correct": pred_overall == overall_gt if isinstance(pred_overall, bool) else None,
    }
    return {
        "per_key": per_key,
        "num_correct": correct,
        "num_evaluated": total,
        "accuracy": correct / total if total else None,
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {key: {"correct": 0, "total": 0} for key in OUTPUT_KEYS}
    totals["overall_nonadherence"] = {"correct": 0, "total": 0}
    for item in items:
        for key, result in item.get("evaluation", {}).get("per_key", {}).items():
            if result.get("correct") is None:
                continue
            totals[key]["total"] += 1
            totals[key]["correct"] += int(result["correct"])
    return {
        key: {
            "accuracy": value["correct"] / value["total"] if value["total"] else None,
            "correct": value["correct"],
            "total": value["total"],
        }
        for key, value in totals.items()
    }


def run_one(
    model: Any,
    processor: Any,
    crop_paths: list[Path],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> str:
    content: list[dict[str, str]] = []
    for idx, path in enumerate(crop_paths):
        content.append({"type": "image", "image": str(path)})
        content.append({"type": "text", "text": f"Image {idx}: view crop from the same clinician."})
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    chat_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if not enable_thinking:
        chat_kwargs["enable_thinking"] = False
    try:
        inputs = processor.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        inputs = processor.apply_chat_template(messages, **chat_kwargs)
    inputs = inputs.to(model.device)
    generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    else:
        generate_kwargs["do_sample"] = False
    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def main() -> None:
    args = parse_args()
    sample_data = load_json(args.samples_json, {})
    ppe_descriptions = load_json(args.ppe_description, {})
    policy = select_policy(load_json(args.policy_set, []), args.requirement_level)
    sample_requirement_level = sample_data.get("requirement_level")
    if sample_requirement_level and sample_requirement_level != args.requirement_level:
        raise ValueError(
            f"samples-json requirement_level={sample_requirement_level} "
            f"does not match --requirement-level={args.requirement_level}"
        )
    image_root = Path(args.image_root) if args.image_root else None
    crop_dir = Path(args.crop_dir)
    samples = sample_items(sample_data["samples"], args.simulation, args.num_samples, args.seed)
    prompt = USER_PROMPT_TEMPLATE.format(
        ppe_descriptions=json.dumps(ppe_descriptions, indent=2),
        requirement_description=policy["requirement_description"],
        violation_visual_cues=json.dumps(policy["violation_visual_cues"], indent=2),
    )

    results = {
        "model": args.model,
        "simulation": args.simulation,
        "num_samples": len(samples),
        "seed": args.seed,
        "requirement_level": args.requirement_level,
        "prediction_schema_version": "alerts_v1",
        "samples_json": args.samples_json,
        "sampling": {
            "method": "random_from_sample_level_json",
            "required_output_categories": OUTPUT_KEYS,
        },
        "items": [],
    }

    model = None
    processor = None
    if not args.sample_only:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        model_kwargs = {
            "dtype": dtype_map[args.dtype],
            "device_map": args.device_map,
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
        processor = AutoProcessor.from_pretrained(args.model)

    for sample_index, sample in enumerate(samples):
        crop_paths = [
            crop_view(sample, view_entry, image_root, crop_dir, sample_index)
            for view_entry in sample["views"]
        ]
        gt = sample["alert_ground_truth"]
        item: dict[str, Any] = {
            "sample_index": sample_index,
            "source_sample_index": sample["index"],
            "sample_id": sample["sample_id"],
            "frame_index": sample["frame_index"],
            "id": sample["clinician_id"],
            "views": [view_entry["view"] for view_entry in sample["views"]],
            "crops": [str(path) for path in crop_paths],
            "alert_ground_truth": gt,
        }
        if model is not None and processor is not None:
            output_text = run_one(
                model,
                processor,
                crop_paths,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.enable_thinking,
            )
            item["raw_output"] = output_text
            try:
                parsed = extract_json(output_text)
                item["prediction"] = parsed
                item["evaluation"] = evaluate_prediction(parsed, gt)
            except Exception as exc:  # noqa: BLE001 - keep raw output for debugging.
                item["parse_error"] = str(exc)
                item["evaluation"] = evaluate_prediction(None, gt)
        results["items"].append(item)
        results["summary"] = summarize(results["items"])
        save_json(args.output, results)
        print(f"saved sample {sample_index} -> {args.output}")


if __name__ == "__main__":
    main()
