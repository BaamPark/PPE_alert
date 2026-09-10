#!/usr/bin/env python3
"""Interactively label cross-view point correspondences and estimate homographies."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FRAME_RE = re.compile(r"^(sim\d+)_view_(\d+)_frame(\d+)\.")


@dataclass
class PointTrack:
    name: str
    views: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class LabelState:
    current_idx: int = 0
    action: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label corresponding physical points across 3 camera views and save "
            "pairwise homographies from a reference view."
        )
    )
    parser.add_argument("--y-json", default="dataset/five_simulation/y1.json")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", default="dataset/five_simulation/homography_gt.json")
    parser.add_argument("--simulation", default="sim1")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--views", default="0,1,2")
    parser.add_argument("--reference-view", type=int, default=0)
    parser.add_argument("--max-panel-width", type=int, default=620)
    parser.add_argument("--max-panel-height", type=int, default=420)
    return parser.parse_args()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def parse_file_name(file_name: str) -> tuple[str, int, int] | None:
    match = FRAME_RE.match(file_name)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def build_frame_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    out: dict[tuple[str, int, int], str] = {}
    for row in rows:
        key = parse_file_name(row["file"])
        if key:
            out.setdefault(key, row["file"])
    return out


def frame_file_name(
    frame_files: dict[tuple[str, int, int], str],
    simulation: str,
    view: int,
    frame_index: int,
) -> str:
    key = (simulation, view, frame_index)
    if key in frame_files:
        return frame_files[key]
    return f"{simulation}_view_{view}_frame{frame_index:03d}.jpg"


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


def toolbar_buttons() -> list[tuple[str, str, tuple[int, int, int, int]]]:
    return [
        ("prev", "PREV", (8, 6, 88, 36)),
        ("next", "NEXT", (96, 6, 176, 36)),
        ("new", "NEW POINT", (184, 6, 324, 36)),
        ("clear", "CLEAR CUR", (332, 6, 472, 36)),
        ("save", "SAVE", (480, 6, 560, 36)),
        ("quit", "QUIT", (568, 6, 648, 36)),
    ]


def toolbar_hit(x: int, y: int) -> str | None:
    for action, _text, (x0, y0, x1, y1) in toolbar_buttons():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return action
    return None


def scale_image(image: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return image.copy(), scale
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))
    return resized, scale


def make_canvas(
    panels: dict[int, np.ndarray],
    panel_offsets: dict[int, tuple[int, int]],
    scales: dict[int, float],
    tracks: list[PointTrack],
    state: LabelState,
    views: list[int],
) -> np.ndarray:
    width = max(x + panels[v].shape[1] for v, (x, _y) in panel_offsets.items()) + 8
    height = max(y + panels[v].shape[0] for v, (_x, y) in panel_offsets.items()) + 8
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for view in views:
        x0, y0 = panel_offsets[view]
        panel = panels[view].copy()
        cv2.putText(panel, f"view {view}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
        cv2.putText(panel, f"view {view}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1)
        canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
        cv2.rectangle(canvas, (x0, y0), (x0 + panel.shape[1], y0 + panel.shape[0]), (90, 90, 90), 2)

    for idx, track in enumerate(tracks):
        active = idx == state.current_idx
        color = (0, 255, 255) if active else (0, 160, 255)
        for view_str, point in track.views.items():
            view = int(view_str)
            if view not in panel_offsets:
                continue
            px = int(round(point[0] * scales[view]))
            py = int(round(point[1] * scales[view]))
            ox, oy = panel_offsets[view]
            cv2.circle(canvas, (ox + px, oy + py), 7 if active else 5, color, -1)
            cv2.putText(
                canvas,
                track.name,
                (ox + px + 8, oy + py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), (0, 0, 0), -1)
    for action, text, (x0, y0, x1, y1) in toolbar_buttons():
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (55, 55, 55), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (210, 210, 210), 1)
        cv2.putText(canvas, text, (x0 + 8, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    current = tracks[state.current_idx] if tracks else None
    status = f"current={current.name if current else '-'} ({state.current_idx + 1}/{len(tracks)}) | click same physical point in each view"
    cv2.putText(canvas, status, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return canvas


def hit_panel(
    x: int,
    y: int,
    panels: dict[int, np.ndarray],
    offsets: dict[int, tuple[int, int]],
) -> tuple[int, int, int] | None:
    for view, panel in panels.items():
        ox, oy = offsets[view]
        h, w = panel.shape[:2]
        if ox <= x < ox + w and oy <= y < oy + h:
            return view, x - ox, y - oy
    return None


def load_existing(output_path: Path, simulation: str, frame_index: int) -> list[PointTrack]:
    data = load_json(output_path, {})
    sim_entry = data.get(simulation, {})
    frame_entry = sim_entry.get("frames", {}).get(str(frame_index), {})
    tracks = []
    for item in frame_entry.get("points", []):
        tracks.append(PointTrack(name=item["name"], views=item.get("views", {})))
    return tracks


def estimate_homographies(
    tracks: list[PointTrack],
    views: list[int],
    reference_view: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    ref_key = str(reference_view)
    for dst_view in views:
        if dst_view == reference_view:
            continue
        dst_key = str(dst_view)
        src_pts = []
        dst_pts = []
        names = []
        for track in tracks:
            if ref_key in track.views and dst_key in track.views:
                src_pts.append(track.views[ref_key])
                dst_pts.append(track.views[dst_key])
                names.append(track.name)
        pair_key = f"{reference_view}_to_{dst_view}"
        item: dict[str, Any] = {"num_points": len(src_pts), "point_names": names}
        if len(src_pts) >= 4:
            src = np.asarray(src_pts, dtype=np.float32)
            dst = np.asarray(dst_pts, dtype=np.float32)
            h_matrix, inlier_mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if h_matrix is not None:
                item["H"] = h_matrix.tolist()
                item["inliers"] = int(inlier_mask.sum()) if inlier_mask is not None else None
        result[pair_key] = item
    return result


def save_output(
    output_path: Path,
    simulation: str,
    frame_index: int,
    views: list[int],
    reference_view: int,
    tracks: list[PointTrack],
) -> None:
    data = load_json(output_path, {})
    sim_entry = data.setdefault(simulation, {"frames": {}})
    frames = sim_entry.setdefault("frames", {})
    frames[str(frame_index)] = {
        "views": views,
        "reference_view": reference_view,
        "points": [{"name": track.name, "views": track.views} for track in tracks],
        "homographies": estimate_homographies(tracks, views, reference_view),
    }
    save_json(output_path, data)


def main() -> None:
    args = parse_args()
    rows = load_json(Path(args.y_json), [])
    image_root = Path(args.image_root) if args.image_root else None
    output_path = Path(args.output)
    views = [int(x) for x in args.views.split(",") if x.strip()]
    frame_files = build_frame_index(rows)

    images = {}
    panels = {}
    scales = {}
    for view in views:
        file_name = frame_file_name(frame_files, args.simulation, view, args.frame_index)
        image = cv2.imread(str(find_image(image_root, file_name)))
        if image is None:
            raise RuntimeError(f"OpenCV failed to read {file_name}")
        images[view] = image
        panels[view], scales[view] = scale_image(image, args.max_panel_width, args.max_panel_height)

    offsets = {}
    x_cursor = 8
    for view in views:
        offsets[view] = (x_cursor, 80)
        x_cursor += panels[view].shape[1] + 8

    tracks = load_existing(output_path, args.simulation, args.frame_index)
    if not tracks:
        tracks = [PointTrack(name="P001")]
    state = LabelState()
    window = "homography_point_labeler"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        action = toolbar_hit(x, y)
        if action == "prev":
            state.current_idx = max(0, state.current_idx - 1)
            return
        if action == "next":
            state.current_idx = min(len(tracks) - 1, state.current_idx + 1)
            return
        if action == "new":
            tracks.append(PointTrack(name=f"P{len(tracks) + 1:03d}"))
            state.current_idx = len(tracks) - 1
            return
        if action == "clear":
            tracks[state.current_idx].views.clear()
            return
        if action in {"save", "quit"}:
            state.action = action
            return

        panel_hit = hit_panel(x, y, panels, offsets)
        if panel_hit is None:
            return
        view, px, py = panel_hit
        orig_x = int(round(px / scales[view]))
        orig_y = int(round(py / scales[view]))
        tracks[state.current_idx].views[str(view)] = [orig_x, orig_y]

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            canvas = make_canvas(panels, offsets, scales, tracks, state, views)
            cv2.imshow(window, canvas)
            if state.action == "save":
                save_output(output_path, args.simulation, args.frame_index, views, args.reference_view, tracks)
                print(f"saved -> {output_path}")
                state.action = None
            if state.action == "quit":
                break
            key = cv2.waitKeyEx(30)
            if key < 0:
                continue
            key = key & 0xFF
            if key in (ord("n"), ord("N")):
                tracks.append(PointTrack(name=f"P{len(tracks) + 1:03d}"))
                state.current_idx = len(tracks) - 1
            elif key in (ord("s"), ord("S")):
                save_output(output_path, args.simulation, args.frame_index, views, args.reference_view, tracks)
                print(f"saved -> {output_path}")
            elif key in (ord("q"), ord("Q")) or key == 27:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
