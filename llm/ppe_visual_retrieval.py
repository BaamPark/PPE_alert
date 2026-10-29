"""Attribute-local, class-diverse retrieval over labeled PPE person crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from archive.fewshot.attribute_recognition.backbones import SigLIP2Backbone


ATTRIBUTES = {
    "gown": ("gown", "the person's shoulders, arms and surgical gown"),
    "mask": ("mask", "the person's nose, mouth and face mask"),
    "eyewear": ("eyewear", "the person's eyes and protective eyewear"),
    "left_glove": ("glove-l", "the person's left hand and glove"),
    "right_glove": ("glove-r", "the person's right hand and glove"),
}

# R4 policy contrast groups. Keep the nearest actual crop from each side;
# the search result is evidence, not the predicted query class.
CONTRAST_LABELS = {
    "gown": (("GC",), ("GI", "GA")),
    "mask": (("NC",), ("RC", "MI", "MA")),
    "eyewear": (("PR", "GG", "FC", "FI", "SG"), ("EA",)),
    "left_glove": (("HC",), ("HA",)),
    "right_glove": (("HC",), ("HA",)),
}

# Person boxes vary by view. The paired proposals keep localization coarse and
# leave the final body-part interpretation to Qwen.
WINDOWS = {
    "gown": ((0.0, 0.05, 1.0, 0.7), (0.0, 0.2, 1.0, 0.95)),
    "mask": ((0.12, 0.0, 0.88, 0.5), (0.05, 0.0, 0.95, 0.65)),
    "eyewear": ((0.1, 0.0, 0.9, 0.4), (0.05, 0.0, 0.95, 0.55)),
    "left_glove": ((0.0, 0.3, 0.6, 1.0), (0.4, 0.3, 1.0, 1.0)),
    "right_glove": ((0.0, 0.3, 0.6, 1.0), (0.4, 0.3, 1.0, 1.0)),
}


@dataclass
class IndexedCrop:
    attribute: str
    label: str
    path: Path
    box: tuple[int, int, int, int]
    feature: torch.Tensor


def proposals(image: Image.Image, attribute: str) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    width, height = image.size
    result = []
    for x0, y0, x1, y1 in WINDOWS[attribute]:
        box = (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
        result.append((image.crop(box), box))
    return result


class PPEVisualRetriever:
    def __init__(self, reference_dir: Path, model_id: str = "google/siglip2-base-patch16-224") -> None:
        if not reference_dir.is_dir():
            raise FileNotFoundError(reference_dir)
        self.backbone = SigLIP2Backbone(model_id=model_id)
        self.index: dict[str, list[IndexedCrop]] = {}
        self._build(reference_dir)

    def _encode_best(self, paths: list[Path], attribute: str) -> list[tuple[tuple[int, int, int, int], torch.Tensor]]:
        crops = []
        boxes = []
        for path in paths:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                candidates = proposals(image, attribute)
                crops.extend(crop for crop, _ in candidates)
                boxes.append([box for _, box in candidates])
        features = self.backbone.encode_images(crops, batch_size=16)
        anchor = self.backbone.encode_texts([ATTRIBUTES[attribute][1]])[0]
        count = len(WINDOWS[attribute])
        result = []
        for idx, candidate_boxes in enumerate(boxes):
            start = idx * count
            best = int((features[start : start + count] @ anchor).argmax())
            result.append((candidate_boxes[best], features[start + best]))
        return result

    def _build(self, reference_dir: Path) -> None:
        for attribute, (folder_prefix, _) in ATTRIBUTES.items():
            labeled_paths = [
                (folder.name[len(folder_prefix) + 1 :], path)
                for folder in sorted(reference_dir.glob(f"{folder_prefix}_*"))
                for path in sorted(folder.glob("*.jpg"))
            ]
            if not labeled_paths:
                raise ValueError(f"No reference images for {attribute} in {reference_dir}")
            selected = self._encode_best([path for _, path in labeled_paths], attribute)
            self.index[attribute] = [
                IndexedCrop(attribute, label, path, box, feature)
                for (label, path), (box, feature) in zip(labeled_paths, selected)
            ]

    def retrieve(self, query_paths: list[Path], attribute: str) -> list[dict]:
        query = self._encode_best(query_paths, attribute)
        candidates = []
        for entry in self.index[attribute]:
            similarities = [float(feature @ entry.feature) for _, feature in query]
            best_view = max(range(len(similarities)), key=similarities.__getitem__)
            candidates.append((similarities[best_view], best_view, entry))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = []
        for labels in CONTRAST_LABELS[attribute]:
            score, view, entry = next(
                item for item in candidates if item[2].label in labels
            )
            selected.append({
                "attribute": attribute,
                "label": entry.label,
                "path": str(entry.path),
                "box": list(entry.box),
                "query_view_index": view,
                "query_box": list(query[view][0]),
                "similarity": score,
            })
        return selected


def save_reference_regions(matches: list[dict], output_dir: Path, size: int = 320) -> list[tuple[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for match in matches:
        source = Path(match["path"])
        with Image.open(source) as opened:
            image = opened.convert("RGB").crop(tuple(match["box"]))
            image.thumbnail((size, size))
            destination = output_dir / f"{match['attribute']}_{match['label']}.jpg"
            image.save(destination, quality=90)
        result.append((f"{match['attribute']}={match['label']}", destination))
    return result
