#!/usr/bin/env python3
"""Build sample-level PPE policy evaluation JSON from row-level annotations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FRAME_RE = re.compile(r"^(sim\d+)_view_(\d+)_frame(\d+)\.")

OUTPUT_TO_ATTRIBUTES = {
    "gown_nonadherence": ["Gown"],
    "mask_nonadherence": ["Mask"],
    "eyewear_nonadherence": ["Eyewear"],
    "left_glove_nonadherence": ["Glove-L"],
    "right_glove_nonadherence": ["Glove-R"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/five_simulation/y1.json")
    parser.add_argument("--policy-set", default="dataset/policy_set.json")
    parser.add_argument("--requirement-level", default="R4")
    parser.add_argument("--meta", default="dataset/five_simulation/meta.json")
    parser.add_argument("--output", default="dataset/five_simulation/y1_policy_eval_samples.json")
    parser.add_argument("--views", default="0,1,2")
    parser.add_argument(
        "--allow-incomplete-views",
        action="store_true",
        help="Keep samples that do not contain all requested views.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")


def parse_file_name(file_name: str) -> tuple[str, int, int]:
    match = FRAME_RE.match(file_name)
    if not match:
        raise ValueError(f"Unexpected file name: {file_name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def select_policy(policy_set: list[dict[str, Any]], requirement_level: str) -> dict[str, Any]:
    for policy in policy_set:
        if policy.get("requirement_level") == requirement_level:
            return policy
    raise KeyError(f"Requirement level not found: {requirement_level}")


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        simulation, view, frame_index = parse_file_name(row["file"])
        item = dict(row)
        item["simulation"] = simulation
        item["view"] = view
        item["frame_index"] = frame_index
        grouped.setdefault((simulation, frame_index, int(row["id"])), []).append(item)
    for items in grouped.values():
        items.sort(key=lambda x: x["view"])
    return grouped


def build_view_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "view": row["view"],
        "file": row["file"],
        "bbox": row["bbox"],
        "view_ground_truth": row["attributes"],
    }


def count_word(count: int) -> str:
    words = {1: "one", 2: "two", 3: "three"}
    return words.get(count, str(count))


def view_word(count: int) -> str:
    return "view" if count == 1 else "views"


def sentence_case_after_prefix(description: str) -> str:
    if not description:
        return description
    return description[0].lower() + description[1:]


def make_evidence_text(
    evidence_items: list[tuple[int, str, str]],
    attribute_descriptions: dict[str, dict[str, str]],
) -> str:
    grouped: dict[tuple[str, str], list[int]] = {}
    for view, attr, value in evidence_items:
        grouped.setdefault((attr, value), []).append(view)

    sentences = []
    for (attr, value), views in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        description = attribute_descriptions.get(attr, {}).get(value, f"The person has {attr}={value}.")
        description = sentence_case_after_prefix(description)
        sentences.append(f"In {count_word(len(views))} {view_word(len(views))}, {description}")
    return " ".join(sentences)


def build_alert_ground_truth(
    rows: list[dict[str, Any]],
    alert_keys: dict[str, list[str]],
    attribute_descriptions: dict[str, dict[str, str]],
) -> dict[str, Any]:
    alert_ground_truth: dict[str, Any] = {}
    for output_key, attrs in OUTPUT_TO_ATTRIBUTES.items():
        evidence_items = []
        alert = False
        papr_present = any(row["attributes"].get("Eyewear") == "PR" for row in rows)
        for row in rows:
            for attr in attrs:
                value = row["attributes"].get(attr)
                if value is None:
                    continue
                evidence_items.append((row["view"], attr, value))
                if value in alert_keys.get(attr, []):
                    alert = True
            if output_key == "mask_nonadherence" and row["attributes"].get("Eyewear") == "PR":
                evidence_items.append((row["view"], "Eyewear", "PR"))
        if output_key == "mask_nonadherence" and papr_present:
            alert = False
        alert_ground_truth[output_key] = {
            "alert": alert,
            "evidence": make_evidence_text(evidence_items, attribute_descriptions),
        }
    alert_ground_truth["overall_nonadherence"] = {
        "alert": any(alert_ground_truth[key]["alert"] for key in OUTPUT_TO_ATTRIBUTES),
        "evidence": "At least one nonadherence alert is true."
        if any(alert_ground_truth[key]["alert"] for key in OUTPUT_TO_ATTRIBUTES)
        else "No nonadherence alerts are true.",
    }
    return alert_ground_truth


def build_samples(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    attribute_descriptions: dict[str, dict[str, str]],
    required_views: list[int],
    allow_incomplete_views: bool,
) -> list[dict[str, Any]]:
    samples = []
    grouped = group_rows(rows)
    for sample_index, ((simulation, frame_index, clinician_id), group) in enumerate(sorted(grouped.items())):
        views_present = {row["view"] for row in group}
        if not allow_incomplete_views and set(required_views) - views_present:
            continue
        selected_rows = [row for row in group if row["view"] in required_views]
        if not selected_rows:
            continue
        sample = {
            "sample_id": f"{simulation}_frame{frame_index:03d}_id{clinician_id}",
            "simulation": simulation,
            "frame_index": frame_index,
            "clinician_id": clinician_id,
            "views": [build_view_entry(row) for row in selected_rows],
            "alert_ground_truth": build_alert_ground_truth(
                selected_rows,
                policy["alert_keys"],
                attribute_descriptions,
            ),
        }
        samples.append(sample)
    for index, sample in enumerate(samples):
        sample["index"] = index
    return samples


def main() -> None:
    args = parse_args()
    rows = load_json(args.input)
    policy = select_policy(load_json(args.policy_set), args.requirement_level)
    meta = load_json(args.meta)
    attribute_descriptions = meta["attributeDescriptions"]
    required_views = [int(x) for x in args.views.split(",") if x.strip()]
    samples = build_samples(rows, policy, attribute_descriptions, required_views, args.allow_incomplete_views)
    output = {
        "source": args.input,
        "requirement_level": args.requirement_level,
        "requirement_description": policy["requirement_description"],
        "schema_version": "policy_eval_samples_v3",
        "alert_semantics": {
            "true": "raise this nonadherence alert",
            "false": "do not raise this nonadherence alert; this may include adherence or insufficient observable evidence",
        },
        "requested_views": required_views,
        "allow_incomplete_views": args.allow_incomplete_views,
        "num_samples": len(samples),
        "output_keys": list(OUTPUT_TO_ATTRIBUTES.keys()) + ["overall_nonadherence"],
        "samples": samples,
    }
    save_json(args.output, output)
    print(f"saved {len(samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()
