#!/usr/bin/env python3
"""Compose a flattened layered-map preview from a base image and prop placements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str, roots: list[Path]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return roots[0] / path


def load_props(data: Any) -> list[dict[str, Any]]:
    """Load placement entries from a list or object with layer lists.

    When the payload is an object, merge *all* of props / foreground / objects
    (in that order) so layered QA previews do not drop platforms or occluders
    just because another key appeared first.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        merged: list[dict[str, Any]] = []
        for key in ("props", "foreground", "objects"):
            value = data.get(key)
            if not isinstance(value, list):
                continue
            if key == "foreground":
                merged.extend({**item, "layer": "foreground"} for item in value)
            else:
                merged.extend(value)
        if merged:
            return merged
    raise ValueError(
        "Placement JSON must be a list or an object with a 'props', "
        "'foreground', and/or 'objects' list."
    )


def placement_xy(prop: dict[str, Any], width: int, height: int) -> tuple[int, int]:
    anchor = str(prop.get("anchor", "center-bottom"))
    x = float(prop.get("x", 0))
    y = float(prop.get("y", 0))

    if anchor == "top-left":
        left = x
        top = y
    elif anchor == "center":
        left = x - width / 2
        top = y - height / 2
    elif anchor == "bottom-left":
        left = x
        top = y - height
    else:
        left = x - width / 2
        top = y - height
    return round(left), round(top)


def paste_prop(canvas: Image.Image, prop: dict[str, Any], roots: list[Path]) -> dict[str, Any]:
    image_key = prop.get("image") or prop.get("path")
    if not image_key:
        raise ValueError(f"Prop is missing image/path: {prop}")
    image_path = resolve_path(str(image_key), roots)
    if not image_path.exists():
        raise FileNotFoundError(f"Prop image not found: {image_path}")

    img = Image.open(image_path).convert("RGBA")
    width = int(prop.get("w", prop.get("width", img.width)))
    height = int(prop.get("h", prop.get("height", img.height)))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid prop size for {image_path}: {width}x{height}")
    if (width, height) != img.size:
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    opacity = float(prop.get("opacity", 1.0))
    if opacity < 1:
        alpha = img.getchannel("A").point(lambda value: int(value * max(0.0, min(1.0, opacity))))
        img.putalpha(alpha)

    left, top = placement_xy(prop, width, height)
    canvas.alpha_composite(img, (left, top))
    return {
        "id": prop.get("id", image_path.stem),
        "image": str(image_path),
        "left": left,
        "top": top,
        "w": width,
        "h": height,
        "sortY": prop.get("sortY", prop.get("y", top + height)),
        "layer": prop.get("layer", "props"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--placements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = Image.open(args.base).convert("RGBA")
    data = read_json(args.placements)
    props = load_props(data)
    roots = [args.placements.parent, args.base.parent, args.project_root]

    props_layer = [prop for prop in props if str(prop.get("layer", "props")) != "foreground"]
    foreground_layer = [prop for prop in props if str(prop.get("layer", "props")) == "foreground"]
    props_layer.sort(key=lambda item: float(item.get("sortY", item.get("y", 0))))
    foreground_layer.sort(key=lambda item: float(item.get("sortY", item.get("y", 0))))

    pasted = []
    for prop in props_layer + foreground_layer:
        pasted.append(paste_prop(base, prop, roots))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    base.save(args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "base": str(args.base),
                    "placements": str(args.placements),
                    "output": str(args.output),
                    "pasted": pasted,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
