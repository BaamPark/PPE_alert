#!/usr/bin/env python3
"""Two-pass Qwen3-VL evaluation with attribute-local SigLIP2 example retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.qwen3vl_policy_nonadherence import (  # noqa: E402
    USER_PROMPT_TEMPLATE,
    crop_view,
    evaluate_prediction,
    extract_json,
    load_json,
    run_one,
    sample_items,
    save_json,
    select_policy,
    summarize,
)
from llm.ppe_visual_retrieval import ATTRIBUTES, PPEVisualRetriever, save_reference_regions  # noqa: E402


PRESENCE_SYSTEM_PROMPT = """Inspect the target clinician across three camera views.
Return valid JSON only."""

PRESENCE_PROMPT = """Look at all three views of the same clinician. Answer each question:
- Is the clinician wearing a gown?
- Is the clinician's face covered by a mask or another item they are wearing?
- Is the clinician wearing protective eyewear or a face shield?
- Is the clinician wearing a glove on the left hand?
- Is the clinician wearing a glove on the right hand?

Use "present", "absent", or "unclear" for each key:
{"gown": "present|absent|unclear", "mask": "present|absent|unclear",
 "eyewear": "present|absent|unclear", "left_glove": "present|absent|unclear",
 "right_glove": "present|absent|unclear"}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--samples-json", default="dataset/five_simulation/y1_policy_eval_samples.json")
    parser.add_argument("--reference-dir", default="archive/fewshot/reference")
    parser.add_argument("--policy-set", default="dataset/policy_set.json")
    parser.add_argument("--ppe-description", default="dataset/ppe_description.json")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--requirement-level", default="R4")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="outputs/llm/qwen3vl_visual_retrieval_R4_50.json")
    parser.add_argument("--crop-dir", default="outputs/llm/qwen3vl_visual_retrieval_R4_50_crops")
    parser.add_argument("--retrieval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requirement_level != "R4":
        raise ValueError("Contrast groups are currently defined only for R4")
    sample_data = load_json(args.samples_json, {})
    if sample_data.get("requirement_level") != args.requirement_level:
        raise ValueError("Sample and requested policy levels differ")
    samples = sample_items(sample_data["samples"], "sim1", args.num_samples, args.seed)
    crop_dir = Path(args.crop_dir)
    image_root = Path(args.image_root) if args.image_root else None
    results = {
        "model": args.model,
        "siglip_model": args.siglip_model,
        "samples_json": args.samples_json,
        "seed": args.seed,
        "reference_dir": args.reference_dir,
        "retrieval": "attribute-local SigLIP2, nearest compliant and violating R4 reference per present/unclear attribute",
        "presence_prompt_version": "simple_presence_v4",
        "items": [],
    }
    for index, sample in enumerate(samples):
        paths = [crop_view(sample, view, image_root, crop_dir, index) for view in sample["views"]]
        results["items"].append({
            "sample_index": index,
            "sample_id": sample["sample_id"],
            "crops": [str(path) for path in paths],
            "alert_ground_truth": sample["alert_ground_truth"],
        })
    save_json(args.output, results)

    # Retrieve before loading Qwen: both backbones otherwise compete for GPU RAM.
    retriever = PPEVisualRetriever(Path(args.reference_dir), args.siglip_model)
    for item in results["items"]:
        paths = [Path(path) for path in item["crops"]]
        item["retrieval"] = {
            attribute: retriever.retrieve(paths, attribute)
            for attribute in ATTRIBUTES
        }
        save_json(args.output, results)
        print(f"retrieved {item['sample_index'] + 1}/{len(samples)}", flush=True)
    del retriever

    if args.retrieval_only:
        return

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.cuda.empty_cache()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    policy = select_policy(load_json(args.policy_set, []), args.requirement_level)
    prompt = USER_PROMPT_TEMPLATE.format(
        ppe_descriptions=json.dumps(load_json(args.ppe_description, {}), indent=2),
        requirement_description=policy["requirement_description"],
        violation_visual_cues=json.dumps(policy["violation_visual_cues"], indent=2),
    )
    prompt += "\nThe labeled references are examples, not the query or a predicted answer. "
    prompt += "Judge the query views directly; a missing reference does not imply adherence."

    for item in results["items"]:
        paths = [Path(path) for path in item["crops"]]
        try:
            presence_raw = run_one(
                model, processor, paths, [], PRESENCE_PROMPT, 128, 0.0, False,
                system_prompt=PRESENCE_SYSTEM_PROMPT,
            )
            try:
                presence = extract_json(presence_raw)
            except (ValueError, TypeError):
                presence = {}
            presence = {
                attribute: presence.get(attribute, "unclear")
                if presence.get(attribute) in {"present", "absent", "unclear"}
                else "unclear"
                for attribute in ATTRIBUTES
            }
            item["presence_raw"] = presence_raw
            item["presence"] = presence
            matches = [
                match
                for attribute in ATTRIBUTES
                if presence[attribute] != "absent"
                for match in item["retrieval"][attribute]
            ]
            references = save_reference_regions(
                matches, crop_dir / f"sample_{item['sample_index']:03d}_{item['sample_id']}" / "retrieved"
            )
            item["selected_references"] = [label for label, _ in references]
            output = run_one(model, processor, paths, references, prompt, args.max_new_tokens, 0.0, False)
            item["raw_output"] = output
            parsed = extract_json(output)
            item["prediction"] = parsed
            item["evaluation"] = evaluate_prediction(parsed, item["alert_ground_truth"])
        except Exception as exc:  # Keep a checkpoint and continue the evaluation.
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["evaluation"] = evaluate_prediction(None, item["alert_ground_truth"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results["summary"] = summarize(results["items"])
        save_json(args.output, results)
        print(f"evaluated {item['sample_index'] + 1}/{len(samples)} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
