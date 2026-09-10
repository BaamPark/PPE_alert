#!/usr/bin/env python3
"""Evaluate cross-view ID matching from GT person boxes and patient anchors."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FRAME_RE = re.compile(r"^(sim\d+)_view_(\d+)_frame(\d+)\.")


@dataclass(frozen=True)
class Detection:
    person_id: int
    view: int
    frame_index: int
    bbox: tuple[float, float, float, float]
    feature: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--y-json", default="dataset/five_simulation/y1.json")
    parser.add_argument("--patient-gt", default="dataset/five_simulation/patient_gt.json")
    parser.add_argument("--simulation", default="sim1")
    parser.add_argument("--views", default="0,1,2")
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=1,
        help="Use the first N frames to choose per-view-pair sign flips. Use 0 for zero-shot.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def parse_file_name(file_name: str) -> tuple[str, int, int]:
    match = FRAME_RE.match(file_name)
    if not match:
        raise ValueError(f"Unexpected file name: {file_name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def patient_basis(anchor: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float], float]:
    head = anchor["patient_head"]
    pelvis = anchor["patient_pelvis"]
    axis_x = pelvis[0] - head[0]
    axis_y = pelvis[1] - head[1]
    length = math.hypot(axis_x, axis_y)
    if length < 1e-6:
        raise ValueError(f"Invalid patient head/pelvis anchor: {anchor}")
    unit_axis = (axis_x / length, axis_y / length)
    unit_perp = (-unit_axis[1], unit_axis[0])
    return unit_axis, unit_perp, length


def patient_center(anchor: dict[str, Any]) -> tuple[float, float]:
    if "patient_obb" in anchor:
        xs = [p[0] for p in anchor["patient_obb"]]
        ys = [p[1] for p in anchor["patient_obb"]]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    x0, y0, x1, y1 = anchor["patient_bbox"]
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def anchor_relative_feature(
    person_bbox: tuple[float, float, float, float],
    anchor: dict[str, Any],
) -> tuple[float, float]:
    cx, cy = bbox_center(person_bbox)
    px, py = patient_center(anchor)
    unit_axis, unit_perp, length = patient_basis(anchor)
    dx = cx - px
    dy = cy - py
    along = (dx * unit_axis[0] + dy * unit_axis[1]) / length
    side = (dx * unit_perp[0] + dy * unit_perp[1]) / length
    return along, side


def build_detections(
    rows: list[dict[str, Any]],
    patient_gt: dict[str, Any],
    simulation: str,
) -> dict[tuple[int, int], list[Detection]]:
    anchors = patient_gt[simulation]["views"]
    grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
    for row in rows:
        sim, view, frame_index = parse_file_name(row["file"])
        if sim != simulation or str(view) not in anchors:
            continue
        bbox = tuple(float(x) for x in row["bbox"])
        feature = anchor_relative_feature(bbox, anchors[str(view)])
        grouped[(frame_index, view)].append(
            Detection(
                person_id=int(row["id"]),
                view=view,
                frame_index=frame_index,
                bbox=bbox,
                feature=feature,
            )
        )
    for detections in grouped.values():
        detections.sort(key=lambda det: det.person_id)
    return grouped


def transform_feature(feature: tuple[float, float], signs: tuple[int, int]) -> tuple[float, float]:
    return signs[0] * feature[0], signs[1] * feature[1]


def squared_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def min_cost_assignment(
    src: list[Detection],
    dst: list[Detection],
    signs: tuple[int, int],
) -> list[tuple[Detection, Detection]]:
    if not src or not dst:
        return []
    if len(src) <= len(dst):
        best_cost = float("inf")
        best_pairs: list[tuple[Detection, Detection]] = []
        for dst_perm in itertools.permutations(dst, len(src)):
            cost = sum(
                squared_distance(s.feature, transform_feature(d.feature, signs))
                for s, d in zip(src, dst_perm)
            )
            if cost < best_cost:
                best_cost = cost
                best_pairs = list(zip(src, dst_perm))
        return best_pairs

    reversed_pairs = min_cost_assignment(dst, src, signs)
    return [(dst_det, src_det) for src_det, dst_det in reversed_pairs]


def evaluate_pair(
    grouped: dict[tuple[int, int], list[Detection]],
    src_view: int,
    dst_view: int,
    frames: list[int],
    signs: tuple[int, int],
) -> tuple[int, int]:
    correct = 0
    total = 0
    for frame in frames:
        src = grouped.get((frame, src_view), [])
        dst = grouped.get((frame, dst_view), [])
        dst_ids = {det.person_id for det in dst}
        pairs = min_cost_assignment(src, dst, signs)
        for src_det, dst_det in pairs:
            if src_det.person_id not in dst_ids:
                continue
            total += 1
            correct += int(src_det.person_id == dst_det.person_id)
    return correct, total


def choose_signs(
    grouped: dict[tuple[int, int], list[Detection]],
    src_view: int,
    dst_view: int,
    frames: list[int],
) -> tuple[int, int]:
    candidates = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    best = (1, 1)
    best_key = (-1.0, -1)
    for signs in candidates:
        correct, total = evaluate_pair(grouped, src_view, dst_view, frames, signs)
        acc = correct / total if total else 0.0
        if (acc, correct) > best_key:
            best_key = (acc, correct)
            best = signs
    return best


def main() -> None:
    args = parse_args()
    rows = load_json(args.y_json)
    patient_gt = load_json(args.patient_gt)
    views = [int(x) for x in args.views.split(",") if x.strip()]
    grouped = build_detections(rows, patient_gt, args.simulation)
    all_frames = sorted({frame for frame, _view in grouped})
    calibration_frames = all_frames[: args.calibration_frames]
    eval_frames = all_frames[args.calibration_frames :] if args.calibration_frames else all_frames

    print(f"simulation={args.simulation}")
    print(f"frames={len(all_frames)} calibration_frames={calibration_frames} eval_frames={len(eval_frames)}")
    print("feature=person bbox center in patient head->pelvis coordinate system")
    print()

    overall_correct = 0
    overall_total = 0
    for src_view, dst_view in itertools.permutations(views, 2):
        signs = (
            choose_signs(grouped, src_view, dst_view, calibration_frames)
            if calibration_frames
            else (1, 1)
        )
        correct, total = evaluate_pair(grouped, src_view, dst_view, eval_frames, signs)
        overall_correct += correct
        overall_total += total
        acc = correct / total if total else 0.0
        print(
            f"view{src_view}->view{dst_view}: "
            f"accuracy={acc:.4f} ({correct}/{total}) signs={signs}"
        )

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    print()
    print(f"overall_accuracy={overall_acc:.4f} ({overall_correct}/{overall_total})")


if __name__ == "__main__":
    main()
