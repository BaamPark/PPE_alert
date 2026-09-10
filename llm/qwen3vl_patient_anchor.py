#!/usr/bin/env python3
"""Run Qwen3-VL patient head/pelvis localization on simulation frames."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


FRAME_RE = re.compile(r"^(sim\d+)_view_(\d+)_frame(\d+)\.")


POINT_PROMPT = """
IMPORTANT: Output must start with "{" and must contain only valid JSON.
Do not explain. Do not think step by step. Do not include markdown.

You are given a camera view of a simulation room with a patient mannequin on a bed.

Locate two anatomical anchor points of the patient mannequin:
1. patient_head: the center point of the patient mannequin's head/face.
2. patient_pelvis: the center point of the patient mannequin's pelvis/groin region.

Return only valid JSON. Use normalized points in the range [0, 1000].
Each point must be [x, y].

JSON schema:
{
  "patient_head": [x, y],
  "patient_pelvis": [x, y]
}
""".strip()


BOX_PROMPT = """
IMPORTANT: Output must start with "{" and must contain only valid JSON.
Do not explain. Do not think step by step. Do not include markdown.

You are given a camera view of a simulation room with a patient mannequin on a bed.

Detect the visible patient mannequin body and the two anatomical anchor regions:
1. patient_box: the visible full patient mannequin body.
2. patient_head: the head/face region of the patient mannequin.
3. patient_pelvis: the pelvis/groin center region of the patient mannequin.

Return only valid JSON. Use normalized bounding boxes in the range [0, 1000].
Each box must be [x1, y1, x2, y2].

JSON schema:
{
  "patient_box": [x1, y1, x2, y2],
  "patient_head": [x1, y1, x2, y2],
  "patient_pelvis": [x1, y1, x2, y2]
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--y-json", default="dataset/five_simulation/y1.json")
    parser.add_argument("--patient-gt", default="dataset/five_simulation/patient_gt.json")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", default="outputs/llm/qwen3vl_patient_points.json")
    parser.add_argument("--overlay-dir", default="outputs/llm/qwen3vl_patient_points_overlay")
    parser.add_argument("--simulation", default="sim1")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--views", default="0,1,2")
    parser.add_argument("--task", choices=["points", "boxes"], default="points")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow Qwen3 thinking mode. Disabled by default because this task needs JSON only.",
    )
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Use flash_attention_2 when the environment supports it.",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Regenerate overlays from an existing --output JSON without model inference.",
    )
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


def parse_file_name(file_name: str) -> tuple[str, int, int] | None:
    match = FRAME_RE.match(file_name)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def build_frame_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    out: dict[tuple[str, int, int], str] = {}
    for row in rows:
        key = parse_file_name(row["file"])
        if key is not None:
            out.setdefault(key, row["file"])
    return out


def frame_file_name(
    frame_files: dict[tuple[str, int, int], str],
    simulation: str,
    view: int,
    frame_index: int,
) -> str:
    return frame_files.get(
        (simulation, view, frame_index),
        f"{simulation}_view_{view}_frame{frame_index:03d}.jpg",
    )


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


def normalized_box_to_pixel(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        x1 / 1000.0 * width,
        y1 / 1000.0 * height,
        x2 / 1000.0 * width,
        y2 / 1000.0 * height,
    ]


def normalized_point_to_pixel(point: list[float], width: int, height: int) -> list[float]:
    x, y = [float(v) for v in point]
    return [x / 1000.0 * width, y / 1000.0 * height]


def box_center(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def point_distance(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def draw_cross(
    draw: ImageDraw.ImageDraw,
    point: list[float],
    color: str,
    radius: int,
    width: int,
) -> None:
    x, y = point
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)


def overlay_style(image: Image.Image) -> tuple[int, int, ImageFont.ImageFont]:
    width, height = image.size
    scale = max(width, height) / 1000.0
    line_width = max(4, int(round(4 * scale)))
    marker_radius = max(12, int(round(10 * scale)))
    font_size = max(28, int(round(22 * scale)))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    return line_width, marker_radius, font


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 5
    bg = (0, 0, 0)
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=bg,
    )
    draw.text((x, y), text, fill=color, font=font)


def save_overlay(
    image_path: Path,
    item: dict[str, Any],
    patient_gt: dict[str, Any],
    simulation: str,
    view: int,
    overlay_dir: str | Path,
) -> str:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width, marker_radius, font = overlay_style(image)
    colors = {
        "patient_box": "lime",
        "patient_head": "deepskyblue",
        "patient_pelvis": "orange",
    }

    prediction_pixel = item.get("prediction_pixel", {})
    draw_boxes = item.get("task") == "boxes"
    for key, pred in prediction_pixel.items():
        if not isinstance(pred, dict):
            continue
        color = colors.get(key, "yellow")
        if draw_boxes and "box_pixel" in pred:
            x1, y1, x2, y2 = pred["box_pixel"]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        center = pred.get("point_pixel") or pred.get("center_pixel")
        if center:
            draw_cross(draw, center, color, marker_radius, line_width)
            draw_label(draw, (center[0] + marker_radius + 4, center[1] - marker_radius), f"pred {key}", color, font)

    gt_view = patient_gt.get(simulation, {}).get("views", {}).get(str(view), {})
    for key in ["patient_head", "patient_pelvis"]:
        gt_point = gt_view.get(key)
        if gt_point:
            draw_cross(draw, gt_point, "red", marker_radius + 5, line_width)
            draw_label(draw, (gt_point[0] + marker_radius + 8, gt_point[1] + 8), f"gt {key}", "red", font)

    out_dir = Path(overlay_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(item['file']).stem}_{item.get('task', 'points')}_overlay.png"
    image.save(out_path)
    return str(out_path)


def run_one(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
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


def convert_prediction(prediction: dict[str, Any], image_size: tuple[int, int], task: str) -> dict[str, Any]:
    width, height = image_size
    converted: dict[str, Any] = {}
    for key, value in prediction.items():
        if not isinstance(value, list):
            converted[key] = {"raw": value, "error": "expected list"}
            continue
        if task == "points":
            if len(value) != 2:
                converted[key] = {"raw": value, "error": "expected [x,y]"}
                continue
            pixel_point = normalized_point_to_pixel(value, width, height)
            converted[key] = {
                "point_norm": value,
                "point_pixel": [round(v, 2) for v in pixel_point],
            }
        else:
            if len(value) != 4:
                converted[key] = {"raw": value, "error": "expected [x1,y1,x2,y2]"}
                continue
            pixel_box = normalized_box_to_pixel(value, width, height)
            converted[key] = {
                "box_norm": value,
                "box_pixel": [round(v, 2) for v in pixel_box],
                "center_pixel": [round(v, 2) for v in box_center(pixel_box)],
            }
    return converted


def add_point_eval(
    item: dict[str, Any],
    patient_gt: dict[str, Any],
    simulation: str,
    view: int,
) -> None:
    gt_view = patient_gt.get(simulation, {}).get("views", {}).get(str(view), {})
    eval_result = {}
    for key in ["patient_head", "patient_pelvis"]:
        pred = item["prediction_pixel"].get(key, {})
        pred_center = pred.get("point_pixel") or pred.get("center_pixel")
        gt_point = gt_view.get(key)
        if pred_center is None or gt_point is None:
            continue
        eval_result[key] = {
            "gt_point": gt_point,
            "pred_center": pred_center,
            "distance_px": round(point_distance(pred_center, gt_point), 2),
        }
    if eval_result:
        item["point_eval"] = eval_result


def main() -> None:
    args = parse_args()
    rows = load_json(args.y_json, [])
    frame_files = build_frame_index(rows)
    patient_gt = load_json(args.patient_gt, {})
    image_root = Path(args.image_root) if args.image_root else None
    views = [int(x) for x in args.views.split(",") if x.strip()]
    prompt = POINT_PROMPT if args.task == "points" else BOX_PROMPT

    if args.overlay_only:
        results = load_json(args.output, {})
        for item in results.get("items", []):
            item["overlay_path"] = save_overlay(
                Path(item["image_path"]),
                item,
                patient_gt,
                args.simulation,
                int(item["view"]),
                args.overlay_dir,
            )
            print(f"saved overlay view {item['view']} -> {item['overlay_path']}")
        save_json(args.output, results)
        return

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

    results = {
        "model": args.model,
        "task": args.task,
        "simulation": args.simulation,
        "frame_index": args.frame_index,
        "output_format": (
            "normalized [x, y], range [0, 1000]"
            if args.task == "points"
            else "normalized [x1, y1, x2, y2], range [0, 1000]"
        ),
        "items": [],
    }

    for view in views:
        file_name = frame_file_name(frame_files, args.simulation, view, args.frame_index)
        image_path = find_image(image_root, file_name)
        with Image.open(image_path) as image:
            image_size = image.size
        output_text = run_one(
            model,
            processor,
            image_path,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.enable_thinking,
        )
        item = {
            "view": view,
            "file": file_name,
            "image_path": str(image_path),
            "image_size": list(image_size),
            "task": args.task,
            "raw_output": output_text,
        }
        try:
            parsed = extract_json(output_text)
            item["prediction_norm"] = parsed
            item["prediction_pixel"] = convert_prediction(parsed, image_size, args.task)
            add_point_eval(item, patient_gt, args.simulation, view)
            item["overlay_path"] = save_overlay(
                image_path,
                item,
                patient_gt,
                args.simulation,
                view,
                args.overlay_dir,
            )
        except Exception as exc:  # noqa: BLE001 - keep raw model output for inspection.
            item["parse_error"] = str(exc)
        results["items"].append(item)
        save_json(args.output, results)
        print(f"saved view {view} -> {args.output}")


if __name__ == "__main__":
    main()
