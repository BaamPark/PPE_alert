#!/usr/bin/env python3
"""Evaluate cross-view ID matching with ground-plane homographies and GT boxes."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FRAME_RE = re.compile(r"^(sim\d+)_view_(\d+)_frame(\d+)\.")


@dataclass(frozen=True)
class Detection:
    person_id: int
    view: int
    frame_index: int
    bbox: tuple[float, float, float, float]
    foot: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="dataset/five_simulation/x1.json")
    parser.add_argument("--homography-gt", default="dataset/five_simulation/homography_gt.json")
    parser.add_argument("--simulation", default=None, help="Defaults to the first simulation in homography_gt.")
    parser.add_argument(
        "--homography-simulation",
        default=None,
        help="Simulation key to read homography from. Defaults to --simulation.",
    )
    parser.add_argument("--homography-frame", type=int, default=0)
    parser.add_argument("--views", default=None, help="Defaults to views stored in homography_gt.")
    parser.add_argument("--threshold", type=float, default=None, help="Reject matched pairs above this pixel distance.")
    parser.add_argument(
        "--mode",
        choices=["target", "reference"],
        default="reference",
        help="target projects source footpoints into the target view; reference maps both views into the reference view.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def parse_file_name(file_name: str) -> tuple[str, int, int]:
    match = FRAME_RE.match(file_name)
    if not match:
        raise ValueError(f"Unexpected file name: {file_name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def bbox_bottom_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, _y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, y1


def build_detections(rows: list[dict[str, Any]], simulation: str, views: set[int]) -> dict[tuple[int, int], list[Detection]]:
    grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
    for row in rows:
        sim, view, frame_index = parse_file_name(row["file"])
        if sim != simulation or view not in views:
            continue
        bbox = tuple(float(x) for x in row["bbox"])
        grouped[(frame_index, view)].append(
            Detection(
                person_id=int(row["id"]),
                view=view,
                frame_index=frame_index,
                bbox=bbox,
                foot=bbox_bottom_center(bbox),
            )
        )
    for detections in grouped.values():
        detections.sort(key=lambda det: det.person_id)
    return grouped


def homography_matrix(homographies: dict[str, Any], reference_view: int, src_view: int, dst_view: int) -> np.ndarray:
    if src_view == dst_view:
        return np.eye(3)

    def ref_to(view: int) -> np.ndarray:
        if view == reference_view:
            return np.eye(3)
        key = f"{reference_view}_to_{view}"
        if key not in homographies or "H" not in homographies[key]:
            raise KeyError(f"Missing homography {key}")
        return np.asarray(homographies[key]["H"], dtype=np.float64)

    h_ref_src = ref_to(src_view)
    h_ref_dst = ref_to(dst_view)
    return h_ref_dst @ np.linalg.inv(h_ref_src)


def project_point(point: tuple[float, float], h_matrix: np.ndarray) -> tuple[float, float]:
    vec = np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    projected = h_matrix @ vec
    if abs(projected[2]) < 1e-12:
        return float("inf"), float("inf")
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def min_cost_assignment(
    src: list[Detection],
    dst: list[Detection],
    src_to_dst: np.ndarray,
    src_to_ref: np.ndarray | None = None,
    dst_to_ref: np.ndarray | None = None,
) -> list[tuple[Detection, Detection, float]]:
    if not src or not dst:
        return []
    if len(src) <= len(dst):
        best_cost = float("inf")
        best_pairs: list[tuple[Detection, Detection, float]] = []
        if src_to_ref is not None and dst_to_ref is not None:
            src_points = {det: project_point(det.foot, src_to_ref) for det in src}
            dst_points = {det: project_point(det.foot, dst_to_ref) for det in dst}
        else:
            src_points = {det: project_point(det.foot, src_to_dst) for det in src}
            dst_points = {det: det.foot for det in dst}
        for dst_perm in itertools.permutations(dst, len(src)):
            costs = [distance(src_points[s], dst_points[d]) for s, d in zip(src, dst_perm)]
            cost = sum(costs)
            if cost < best_cost:
                best_cost = cost
                best_pairs = [(s, d, c) for s, d, c in zip(src, dst_perm, costs)]
        return best_pairs

    reversed_pairs = min_cost_assignment(dst, src, np.linalg.inv(src_to_dst), dst_to_ref, src_to_ref)
    return [(dst_det, src_det, cost) for src_det, dst_det, cost in reversed_pairs]


def evaluate_pair(
    grouped: dict[tuple[int, int], list[Detection]],
    frames: list[int],
    src_view: int,
    dst_view: int,
    src_to_dst: np.ndarray,
    threshold: float | None,
    src_to_ref: np.ndarray | None = None,
    dst_to_ref: np.ndarray | None = None,
) -> tuple[int, int, int]:
    correct = 0
    total = 0
    rejected = 0
    for frame in frames:
        src = grouped.get((frame, src_view), [])
        dst = grouped.get((frame, dst_view), [])
        dst_ids = {det.person_id for det in dst}
        for src_det, dst_det, cost in min_cost_assignment(src, dst, src_to_dst, src_to_ref, dst_to_ref):
            if src_det.person_id not in dst_ids:
                continue
            if threshold is not None and cost > threshold:
                rejected += 1
                continue
            total += 1
            correct += int(src_det.person_id == dst_det.person_id)
    return correct, total, rejected


def main() -> None:
    args = parse_args()
    rows = load_json(args.json)
    h_gt = load_json(args.homography_gt)
    simulation = args.simulation or sorted(h_gt.keys())[0]
    homography_simulation = args.homography_simulation or simulation
    frame_entry = h_gt[homography_simulation]["frames"][str(args.homography_frame)]
    views = [int(x) for x in args.views.split(",")] if args.views else [int(v) for v in frame_entry["views"]]
    reference_view = int(frame_entry["reference_view"])
    homographies = frame_entry["homographies"]

    grouped = build_detections(rows, simulation, set(views))
    frames = sorted({frame for frame, _view in grouped})

    print(f"simulation={simulation}")
    print(f"homography_simulation={homography_simulation}")
    print(f"frames={len(frames)} homography_frame={args.homography_frame} reference_view={reference_view}")
    print(f"feature=bbox bottom-center projected by ground-plane homography threshold={args.threshold} mode={args.mode}")
    print()

    overall_correct = 0
    overall_total = 0
    overall_rejected = 0
    for src_view, dst_view in itertools.permutations(views, 2):
        src_to_dst = homography_matrix(homographies, reference_view, src_view, dst_view)
        if args.mode == "reference":
            src_to_ref = homography_matrix(homographies, reference_view, src_view, reference_view)
            dst_to_ref = homography_matrix(homographies, reference_view, dst_view, reference_view)
        else:
            src_to_ref = None
            dst_to_ref = None
        correct, total, rejected = evaluate_pair(
            grouped,
            frames,
            src_view,
            dst_view,
            src_to_dst,
            args.threshold,
            src_to_ref,
            dst_to_ref,
        )
        overall_correct += correct
        overall_total += total
        overall_rejected += rejected
        acc = correct / total if total else 0.0
        print(
            f"view{src_view}->view{dst_view}: "
            f"accuracy={acc:.4f} ({correct}/{total}) rejected={rejected}"
        )

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    print()
    print(f"overall_accuracy={overall_acc:.4f} ({overall_correct}/{overall_total}) rejected={overall_rejected}")


if __name__ == "__main__":
    main()
