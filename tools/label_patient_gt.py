#!/usr/bin/env python3
"""Interactively label static patient anchors for each simulation view."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2


FRAME_RE = re.compile(r"^(sim\d+)_view_(\d+)_frame(\d+)\.")


@dataclass
class ViewLabel:
    patient_head: list[int] | None = None
    patient_pelvis: list[int] | None = None


@dataclass
class LabelState:
    mode: str = "head"
    action: str | None = None
    labels: dict[str, ViewLabel] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label patient head/pelvis once per simulation view. "
            "The labels are static and apply to all frame indices in that view."
        )
    )
    parser.add_argument("--y-json", default="dataset/five_simulation/y1.json")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Directory containing frame images. If omitted, common local dataset paths are checked.",
    )
    parser.add_argument("--output", default="dataset/five_simulation/patient_gt.json")
    parser.add_argument("--simulation", default="sim1", help="Example: sim1")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--views", default="0,1,2")
    return parser.parse_args()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def image_key(file_name: str) -> tuple[str, int, int] | None:
    match = FRAME_RE.match(file_name)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def build_frame_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    out: dict[tuple[str, int, int], str] = {}
    for row in rows:
        key = image_key(row["file"])
        if key is not None:
            out.setdefault(key, row["file"])
    return out


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
    raise FileNotFoundError(
        f"Could not find {file_name}. Searched: {searched}\n"
        "Pass the real image directory with --image-root, for example:\n"
        "  --image-root /home/beomseok/Datasets/five_simulation/images"
    )

def person_boxes_for_frame(
    rows: list[dict[str, Any]], simulation: str, view: int, frame_index: int
) -> list[list[int]]:
    boxes = []
    for row in rows:
        key = image_key(row["file"])
        if key == (simulation, view, frame_index):
            boxes.append(row["bbox"])
    return boxes


def toolbar_buttons() -> list[tuple[str, str, tuple[int, int, int, int]]]:
    return [
        ("head", "HEAD", (8, 6, 88, 36)),
        ("pelvis", "PELVIS", (96, 6, 196, 36)),
        ("reset", "RESET", (204, 6, 304, 36)),
        ("save", "SAVE/NEXT", (312, 6, 452, 36)),
        ("quit", "QUIT", (460, 6, 540, 36)),
    ]


def toolbar_hit(x: int, y: int) -> str | None:
    for action, _label, (x0, y0, x1, y1) in toolbar_buttons():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return action
    return None


def draw_overlay(
    image: Any,
    view: int,
    label: ViewLabel,
    state: LabelState,
    person_boxes: list[list[int]],
) -> Any:
    canvas = image.copy()
    for box in person_boxes:
        x0, y0, x1, y1 = box
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (170, 170, 170), 2)

    if label.patient_head:
        cv2.circle(canvas, tuple(label.patient_head), 8, (255, 0, 0), -1)
        cv2.putText(canvas, "head", tuple(label.patient_head), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if label.patient_pelvis:
        cv2.circle(canvas, tuple(label.patient_pelvis), 8, (0, 180, 255), -1)
        cv2.putText(canvas, "pelvis", tuple(label.patient_pelvis), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), (0, 0, 0), -1)
    for action, text, (x0, y0, x1, y1) in toolbar_buttons():
        active = action == state.mode
        color = (50, 130, 255) if active else (55, 55, 55)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (210, 210, 210), 1)
        cv2.putText(canvas, text, (x0 + 8, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    status = f"view={view} mode={state.mode} | click HEAD/PELVIS points; keyboard: h/p/r/s/q"
    cv2.putText(canvas, status, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return canvas


def upsert_output(output_path: Path, simulation: str, labels: dict[str, ViewLabel]) -> None:
    data = load_json(output_path, {})
    sim_entry = data.setdefault(simulation, {"views": {}})
    views = sim_entry.setdefault("views", {})
    for view, label in labels.items():
        item = {}
        if label.patient_head:
            item["patient_head"] = label.patient_head
        if label.patient_pelvis:
            item["patient_pelvis"] = label.patient_pelvis
        views[str(view)] = item
    save_json(output_path, data)


def view_label_from_json(item: dict[str, Any]) -> ViewLabel:
    if "patient_feet" in item and "patient_pelvis" not in item:
        item = {**item, "patient_pelvis": item["patient_feet"]}
    return ViewLabel(
        patient_head=item.get("patient_head"),
        patient_pelvis=item.get("patient_pelvis"),
    )


def label_view(window: str, image: Any, view: int, label: ViewLabel, person_boxes: list[list[int]]) -> ViewLabel:
    state = LabelState(labels={str(view): label})

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        current = state.labels[str(view)]
        if event == cv2.EVENT_LBUTTONDOWN:
            action = toolbar_hit(x, y)
            if action in {"head", "pelvis"}:
                state.mode = action
                return
            if action == "reset":
                state.labels[str(view)] = ViewLabel()
                return
            if action in {"save", "quit"}:
                state.action = action
                return
        if event == cv2.EVENT_LBUTTONDOWN and state.mode == "head":
            current.patient_head = [x, y]
        elif event == cv2.EVENT_LBUTTONDOWN and state.mode == "pelvis":
            current.patient_pelvis = [x, y]

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        current = state.labels[str(view)]
        cv2.imshow(window, draw_overlay(image, view, current, state, person_boxes))
        if state.action == "save":
            return state.labels[str(view)]
        if state.action == "quit":
            raise KeyboardInterrupt
        key = cv2.waitKeyEx(30)
        if key < 0:
            continue
        key = key & 0xFF
        if key in (ord("h"), ord("H")):
            state.mode = "head"
        elif key in (ord("p"), ord("P")):
            state.mode = "pelvis"
        elif key in (ord("r"), ord("R")):
            state.labels[str(view)] = ViewLabel()
        elif key in (ord("n"), ord("s"), ord("N"), ord("S")):
            return state.labels[str(view)]
        elif key in (ord("q"), ord("Q")) or key == 27:
            raise KeyboardInterrupt


def main() -> None:
    args = parse_args()
    y_path = Path(args.y_json)
    image_root = Path(args.image_root) if args.image_root else None
    output_path = Path(args.output)
    rows = load_json(y_path, [])
    frame_files = build_frame_index(rows)
    views = [int(x) for x in args.views.split(",") if x.strip()]

    existing = load_json(output_path, {})
    existing_views = existing.get(args.simulation, {}).get("views", {})
    labels = {
        str(view): view_label_from_json(existing_views.get(str(view), {}))
        for view in views
    }

    window = "patient_gt_labeler"
    try:
        for view in views:
            file_name = frame_files[(args.simulation, view, args.frame_index)]
            image_path = find_image(image_root, file_name)
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"OpenCV failed to read {image_path}")
            boxes = person_boxes_for_frame(rows, args.simulation, view, args.frame_index)
            labels[str(view)] = label_view(window, image, view, labels[str(view)], boxes)
            upsert_output(output_path, args.simulation, labels)
            print(f"saved {args.simulation} view {view} -> {output_path}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
