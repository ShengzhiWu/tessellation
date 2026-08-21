#!/usr/bin/env python3
"""Visualize hierarchical hat patches using Tile(a,b) and reflected copies.

This script renders the hat case, Tile(1,sqrt(3)), from the alternative
substitution system in Figure 2.11 of the first aperiodic monotile paper.  The
source SVG contains H_8, followed by two further H_8 iterations.  Compound
tiles are used internally by the substitution and split back into one hat and
one reflected hat for the output.
"""

from __future__ import annotations

import argparse
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


A = 1.0
B = math.sqrt(3.0)
SOURCE_URL = "https://arxiv.org/html/2303.10798v3/alt_subst.svg"
SOURCE_PATH = Path("references/alt_subst.svg")

REFLECTED_FILL = "#0089d4"
SOURCE_FILLS = {"#fafafa", "#94cdeb", "#0089d4"}

COLORS = {
    "unreflected": "#d9b487",
    "reflected": "#8ca8c8",
}


@dataclass(frozen=True)
class ShapePath:
    fill: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class HatPolygon:
    reflected: bool
    points: tuple[tuple[float, float], ...]


def parse_transform(value: str) -> tuple[float, float, float, float, float, float]:
    match = re.fullmatch(r"matrix\(([^)]+)\)", value.strip())
    if not match:
        raise ValueError(f"Unsupported transform: {value}")
    parts = [float(part) for part in re.split(r"[,\s]+", match.group(1).strip()) if part]
    if len(parts) != 6:
        raise ValueError(f"Expected six matrix entries in: {value}")
    return tuple(parts)  # type: ignore[return-value]


def parse_path_points(d: str) -> tuple[tuple[float, float], ...]:
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d+\.\d*|\.\d+|\d+)", d)
    points: list[tuple[float, float]] = []
    x = 0.0
    y = 0.0
    i = 0
    command = ""

    while i < len(tokens):
        token = tokens[i]
        if re.fullmatch(r"[A-Za-z]", token):
            command = token
            i += 1
            if command.upper() == "Z":
                continue

        if command == "M":
            x = float(tokens[i])
            y = float(tokens[i + 1])
            points.append((x, y))
            i += 2
            command = "L"
        elif command == "m":
            x += float(tokens[i])
            y += float(tokens[i + 1])
            points.append((x, y))
            i += 2
            command = "l"
        elif command == "L":
            x = float(tokens[i])
            y = float(tokens[i + 1])
            points.append((x, y))
            i += 2
        elif command == "l":
            x += float(tokens[i])
            y += float(tokens[i + 1])
            points.append((x, y))
            i += 2
        elif command == "H":
            x = float(tokens[i])
            points.append((x, y))
            i += 1
        elif command == "h":
            x += float(tokens[i])
            points.append((x, y))
            i += 1
        elif command == "V":
            y = float(tokens[i])
            points.append((x, y))
            i += 1
        elif command == "v":
            y += float(tokens[i])
            points.append((x, y))
            i += 1
        else:
            raise ValueError(f"Unsupported SVG path command {command!r} in {d[:80]}...")

    if points and points[-1] == points[0]:
        points.pop()
    if len(points) < 3:
        raise ValueError(f"Path does not describe a polygon: {d[:80]}...")
    return tuple(points)


def transform_point(
    matrix: tuple[float, float, float, float, float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    x, y = point
    return (a * x + c * y + e, b * x + d * y + f)


def polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        / 2.0
    )


def centroid(points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> tuple[float, float]:
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )


def bounds(points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
    return (
        min(x for x, _ in points),
        min(y for _, y in points),
        max(x for x, _ in points),
        max(y for _, y in points),
    )


def affine_from_tri(
    source: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    target: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[float, float, float, float, float, float]:
    (x0, y0), (x1, y1), (x2, y2) = source
    (u0, v0), (u1, v1), (u2, v2) = target
    sx1 = x1 - x0
    sy1 = y1 - y0
    sx2 = x2 - x0
    sy2 = y2 - y0
    tx1 = u1 - u0
    ty1 = v1 - v0
    tx2 = u2 - u0
    ty2 = v2 - v0
    det = sx1 * sy2 - sx2 * sy1
    if abs(det) < 1e-9:
        raise ValueError("Source triangle is degenerate")

    a = (tx1 * sy2 - tx2 * sy1) / det
    c = (sx1 * tx2 - sx2 * tx1) / det
    b = (ty1 * sy2 - ty2 * sy1) / det
    d = (sx1 * ty2 - sx2 * ty1) / det
    e = u0 - a * x0 - c * y0
    f = v0 - b * x0 - d * y0
    return (a, b, c, d, e, f)


def match_affine(
    source: tuple[tuple[float, float], ...],
    target: tuple[tuple[float, float], ...],
    tolerance: float = 0.2,
) -> tuple[float, float, float, float, float, float]:
    if len(source) != len(target):
        raise ValueError("Cannot match polygons with different vertex counts")

    tri_index = 2
    while tri_index < len(source):
        area = abs(
            (source[1][0] - source[0][0]) * (source[tri_index][1] - source[0][1])
            - (source[1][1] - source[0][1]) * (source[tri_index][0] - source[0][0])
        )
        if area > 1e-6:
            break
        tri_index += 1
    if tri_index == len(source):
        raise ValueError("Source polygon is degenerate")

    for reversed_order in (False, True):
        sequence = tuple(reversed(target)) if reversed_order else target
        for offset in range(len(target)):
            candidate = sequence[offset:] + sequence[:offset]
            matrix = affine_from_tri(
                (source[0], source[1], source[tri_index]),
                (candidate[0], candidate[1], candidate[tri_index]),
            )
            error = max(
                math.dist(transform_point(matrix, point), expected)
                for point, expected in zip(source, candidate)
            )
            if error <= tolerance:
                return matrix

    raise ValueError("Could not match compound outline to canonical compound")


def ensure_source(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        path.write_bytes(response.read())


def load_source_shapes(source: Path) -> list[ShapePath]:
    ensure_source(source)
    root = ET.parse(source).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    shapes: list[ShapePath] = []

    for path in root.iter(namespace + "path"):
        fill = (path.get("fill") or "").lower()
        if fill not in SOURCE_FILLS:
            continue
        d = path.get("d")
        transform = path.get("transform")
        if not d or not transform:
            continue
        local_points = parse_path_points(d)
        matrix = parse_transform(transform)
        points = tuple(transform_point(matrix, point) for point in local_points)
        if len(points) < 13 or polygon_area(points) > 1000:
            continue
        shapes.append(ShapePath(fill, points))

    return shapes


def connected_components(shapes: list[ShapePath]) -> list[list[ShapePath]]:
    try:
        from shapely.geometry import Polygon
    except ModuleNotFoundError as exc:
        raise SystemExit("This script requires shapely in the active Python environment.") from exc

    polygons = [Polygon(shape.points) for shape in shapes]
    adjacency: list[list[int]] = [[] for _ in shapes]
    for i, polygon in enumerate(polygons):
        for j in range(i + 1, len(polygons)):
            if polygon.distance(polygons[j]) < 0.02:
                adjacency[i].append(j)
                adjacency[j].append(i)

    seen: set[int] = set()
    components: list[list[ShapePath]] = []
    for start in range(len(shapes)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[ShapePath] = []
        while stack:
            index = stack.pop()
            component.append(shapes[index])
            for neighbor in adjacency[index]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    return sorted(components, key=len)


def component_bounds(component: list[ShapePath]) -> tuple[float, float, float, float]:
    all_points = [point for shape in component for point in shape.points]
    return bounds(all_points)


def load_hierarchy(source: Path) -> tuple[dict[int, list[ShapePath]], ShapePath, list[ShapePath]]:
    shapes = load_source_shapes(source)
    components = connected_components(shapes)

    single_hats = [
        component[0]
        for component in components
        if len(component) == 1 and len(component[0].points) == 13 and component[0].fill != REFLECTED_FILL
    ]
    compounds = [
        component[0]
        for component in components
        if len(component) == 1 and len(component[0].points) == 19
    ]
    split_compounds = [
        component
        for component in components
        if len(component) == 2 and all(len(shape.points) == 13 for shape in component)
    ]

    if not single_hats or not compounds or not split_compounds:
        raise SystemExit("Could not identify H, compound, and split-compound prototypes in the source SVG.")

    levels: dict[int, list[ShapePath]] = {0: [single_hats[0]]}
    for depth, size in ((1, 7), (2, 48), (3, 329)):
        candidates = [component for component in components if len(component) == size]
        if not candidates:
            raise SystemExit(f"Could not identify the n={depth} H_8 iteration in the source SVG.")
        levels[depth] = sorted(candidates, key=lambda component: component_bounds(component)[0])[0]

    return levels, compounds[0], split_compounds[0]


def expand_compounds(
    shapes: list[ShapePath],
    canonical_compound: ShapePath,
    split_compound: list[ShapePath],
) -> list[HatPolygon]:
    compound_min_x, compound_min_y, _, _ = bounds(canonical_compound.points)
    split_min_x, split_min_y, _, _ = bounds([point for shape in split_compound for point in shape.points])
    shift = (compound_min_x - split_min_x, compound_min_y - split_min_y)
    shifted_split = [
        ShapePath(
            shape.fill,
            tuple((x + shift[0], y + shift[1]) for x, y in shape.points),
        )
        for shape in split_compound
    ]

    polygons: list[HatPolygon] = []
    for shape in shapes:
        if len(shape.points) == 19:
            matrix = match_affine(canonical_compound.points, shape.points)
            for split_shape in shifted_split:
                polygons.append(
                    HatPolygon(
                        split_shape.fill == REFLECTED_FILL,
                        tuple(transform_point(matrix, point) for point in split_shape.points),
                    )
                )
        else:
            polygons.append(HatPolygon(False, shape.points))
    return polygons


def draw_svg(polygons: list[HatPolygon], output: Path, labels: bool) -> None:
    all_points = [point for polygon in polygons for point in polygon.points]
    min_x, min_y, max_x, max_y = bounds(all_points)
    pad = 10.0
    view_box = (min_x - pad, min_y - pad, max_x - min_x + 2 * pad, max_y - min_y + 2 * pad)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box[0]:.3f} {view_box[1]:.3f} '
            f'{view_box[2]:.3f} {view_box[3]:.3f}" width="1400">\n'
        )
        handle.write('<rect x="-100000" y="-100000" width="200000" height="200000" fill="#fbfaf6"/>\n')
        handle.write('<g stroke="#17202a" stroke-width="0.45" stroke-linejoin="round" stroke-linecap="round">\n')
        for polygon in polygons:
            color = COLORS["reflected"] if polygon.reflected else COLORS["unreflected"]
            point_text = " ".join(f"{x:.3f},{y:.3f}" for x, y in polygon.points)
            handle.write(f'<polygon points="{point_text}" fill="{color}"/>\n')
        handle.write("</g>\n")

        if labels:
            handle.write('<g font-family="Arial, sans-serif" font-size="5" fill="#17202a">\n')
            for polygon in polygons:
                if not polygon.reflected:
                    continue
                cx, cy = centroid(polygon.points)
                handle.write(
                    f'<text x="{cx:.3f}" y="{cy:.3f}" text-anchor="middle" '
                    'dominant-baseline="central">R</text>\n'
                )
            handle.write("</g>\n")

        handle.write("</svg>\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize hierarchical hat + reflected-hat substitution patches.")
    parser.add_argument("-n", "--iterations", type=int, default=3, help="H_8 substitution depth; supported range is 0..3")
    parser.add_argument("--a", type=float, default=A, help="must be 1 for this source patch")
    parser.add_argument("--b", type=float, default=B, help="must be sqrt(3) for this source patch")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/hat.svg"), help="output SVG path")
    parser.add_argument("--source", type=Path, default=SOURCE_PATH, help="source alt_subst.svg path; downloaded from arXiv if missing")
    parser.add_argument("--labels", action="store_true", help="mark reflected hats with R")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(args.a - A) > 1e-9 or abs(args.b - B) > 1e-9:
        raise SystemExit("This source substitution is currently available only for Tile(1,sqrt(3)).")
    if args.iterations < 0 or args.iterations > 3:
        raise SystemExit("Only n=0..3 are available from the Figure 2.11 source SVG.")

    levels, canonical_compound, split_compound = load_hierarchy(args.source)
    polygons = expand_compounds(levels[args.iterations], canonical_compound, split_compound)
    reflected = sum(1 for polygon in polygons if polygon.reflected)
    draw_svg(polygons, args.output, args.labels)
    print(f"Wrote {args.output} with {len(polygons)} hats ({reflected} reflected).")


if __name__ == "__main__":
    main()
