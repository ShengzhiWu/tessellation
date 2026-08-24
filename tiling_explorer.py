#!/usr/bin/env python3
"""Explore polygon tilings by depth-first boundary growth.

The search starts with one tile, then repeatedly chooses an exposed boundary
edge of the current patch and tries to attach a fresh copy of the input polygon
by making one of its edges collinear in the opposite direction and sharing one
endpoint.  Accepted DFS states can be exported as PNGs at a fixed interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point as ShapelyPoint, Polygon
from shapely.ops import snap, unary_union


SQRT3 = math.sqrt(3.0)


Point = tuple[float, float]


@dataclass(frozen=True)
class Tile:
    points: tuple[Point, ...]
    reflected: bool = False


@dataclass(frozen=True)
class State:
    tiles: tuple[Tile, ...]
    depth: int
    path_keys: frozenset[tuple[tuple[tuple[float, float], ...], ...]]


def preset_polygon(name: str) -> tuple[Point, ...]:
    if name == "square":
        return ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    if name == "tile11":
        return (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.5, -SQRT3 / 2.0),
            (1.5 + SQRT3 / 2.0, 0.5 - SQRT3 / 2.0),
            (1.5 + SQRT3 / 2.0, 1.5 - SQRT3 / 2.0),
            (2.5 + SQRT3 / 2.0, 1.5 - SQRT3 / 2.0),
            (3.0 + SQRT3 / 2.0, 1.5),
            (3.0, 2.0),
            (3.0 - SQRT3 / 2.0, 1.5),
            (2.5 - SQRT3 / 2.0, 1.5 + SQRT3 / 2.0),
            (1.5 - SQRT3 / 2.0, 1.5 + SQRT3 / 2.0),
            (0.5 - SQRT3 / 2.0, 1.5 + SQRT3 / 2.0),
            (-SQRT3 / 2.0, 1.5),
            (0.0, 1.0),
        )

    if name == "hat":
        def hex_point(x: float, y: float) -> Point:
            return (x + 0.5 * y, -SQRT3 * y / 2.0)

        return tuple(
            hex_point(x, y)
            for x, y in (
                (-1, 2),
                (0, 2),
                (0, 3),
                (2, 2),
                (3, 0),
                (4, 0),
                (5, -1),
                (4, -2),
                (2, -1),
                (2, -2),
                (1, -2),
                (0, -2),
                (-1, -1),
                (0, 0),
            )
        )

    raise ValueError(f"Unknown preset: {name}")


def load_polygon(path: Path) -> tuple[Point, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 3:
        raise SystemExit("Polygon JSON must be a list of at least three [x, y] points.")

    points = []
    for item in data:
        if not isinstance(item, list) or len(item) != 2:
            raise SystemExit("Polygon JSON points must have the form [x, y].")
        points.append((float(item[0]), float(item[1])))
    return tuple(points)


def remove_collinear_vertices(points: tuple[Point, ...], tolerance: float = 1e-9) -> tuple[Point, ...]:
    cleaned = list(points)
    changed = True
    while changed and len(cleaned) > 3:
        changed = False
        next_points: list[Point] = []
        for index, point in enumerate(cleaned):
            previous = cleaned[index - 1]
            following = cleaned[(index + 1) % len(cleaned)]
            incoming = (point[0] - previous[0], point[1] - previous[1])
            outgoing = (following[0] - point[0], following[1] - point[1])
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            scale = math.hypot(*incoming) * math.hypot(*outgoing)
            if scale > 0 and abs(cross) <= tolerance * scale and dot > 0:
                changed = True
                continue
            next_points.append(point)
        cleaned = next_points
    return tuple(cleaned)


def normalize_base(points: tuple[Point, ...]) -> tuple[Point, ...]:
    points = remove_collinear_vertices(points)
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 0:
        raise SystemExit("Input polygon must be valid and have positive area.")

    cx, cy = polygon.centroid.x, polygon.centroid.y
    return tuple((x - cx, y - cy) for x, y in points)


def edges(points: tuple[Point, ...]) -> Iterable[tuple[Point, Point]]:
    yield from zip(points, points[1:] + points[:1])


def point_diameter(points: tuple[Point, ...]) -> float:
    return max(math.dist(a, b) for index, a in enumerate(points) for b in points[index + 1 :])


def parse_length_value(text: str) -> float:
    normalized = text.strip().lower().replace(" ", "")
    if normalized in {"sqrt3", "sqrt(3)", "√3"}:
        return SQRT3
    return float(normalized)


def parse_allowed_length_pairs(text: str | None) -> tuple[tuple[float, float], ...] | None:
    if text is None or not text.strip():
        return None

    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        separator = ":" if ":" in item else "-"
        parts = item.split(separator)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("length pairs must look like 1:1,1:2,sqrt3:sqrt3")
        a = parse_length_value(parts[0])
        b = parse_length_value(parts[1])
        pairs.append(tuple(sorted((a, b))))
    if not pairs:
        raise argparse.ArgumentTypeError("at least one length pair is required")
    return tuple(pairs)


def length_pair_allowed(
    first: float,
    second: float,
    allowed_pairs: tuple[tuple[float, float], ...] | None,
    tolerance: float,
) -> bool:
    if allowed_pairs is None:
        return True

    pair = tuple(sorted((first, second)))
    return any(abs(pair[0] - allowed[0]) <= tolerance and abs(pair[1] - allowed[1]) <= tolerance for allowed in allowed_pairs)


def rotate_point(point: Point, angle: float) -> Point:
    x, y = point
    c = math.cos(angle)
    s = math.sin(angle)
    return (c * x - s * y, s * x + c * y)


def transform_points(points: tuple[Point, ...], reflected: bool, angle: float, dx: float, dy: float) -> tuple[Point, ...]:
    reflected_points = tuple((-x, y) for x, y in points) if reflected else points
    return tuple((rx + dx, ry + dy) for rx, ry in (rotate_point(point, angle) for point in reflected_points))


def boundary_segments(patch) -> list[tuple[Point, Point]]:
    raw_segments: list[tuple[Point, Point]] = []

    def add_line(line: LineString) -> None:
        coords = list(line.coords)
        for a, b in zip(coords, coords[1:]):
            if math.dist(a, b) > 1e-8:
                raw_segments.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))

    boundary = patch.boundary
    if isinstance(boundary, LineString):
        add_line(boundary)
    elif isinstance(boundary, MultiLineString):
        for line in boundary.geoms:
            add_line(line)
    elif isinstance(boundary, GeometryCollection):
        for geom in boundary.geoms:
            if isinstance(geom, LineString):
                add_line(geom)
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    add_line(line)

    segments: list[tuple[Point, Point]] = []
    for a, b in raw_segments:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= 1e-8:
            continue

        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        normal = (-dy / length, dx / length)
        probe = min(1e-4, length * 1e-4)
        left = ShapelyPoint(mid[0] + normal[0] * probe, mid[1] + normal[1] * probe)
        right = ShapelyPoint(mid[0] - normal[0] * probe, mid[1] - normal[1] * probe)

        if patch.covers(left) == patch.covers(right):
            continue
        segments.append((a, b))

    return segments


def boundary_segments_by_interior_score(
    patch,
    samples: int,
    probe_radius: float,
) -> list[tuple[float, tuple[Point, Point]]]:
    from shapely.geometry import Point as ShapelyPoint

    cache: dict[tuple[float, float], float] = {}

    def filled_angle(point: Point) -> float:
        key = (round(point[0], 8), round(point[1], 8))
        if key in cache:
            return cache[key]

        hits = 0
        for index in range(samples):
            angle = 2.0 * math.pi * index / samples
            probe = ShapelyPoint(
                point[0] + probe_radius * math.cos(angle),
                point[1] + probe_radius * math.sin(angle),
            )
            if patch.covers(probe):
                hits += 1
        value = 2.0 * math.pi * hits / samples
        cache[key] = value
        return value

    scored = []
    for segment in boundary_segments(patch):
        score = filled_angle(segment[0]) + filled_angle(segment[1])
        scored.append((score, segment))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def boundary_segments_by_bitmap_score(
    patch,
    tile_diameter: float,
    pixels_per_tile_diameter: float,
    blur_radius_diameters: float,
) -> list[tuple[float, tuple[Point, Point]]]:
    segments = boundary_segments(patch)
    if not segments:
        return []

    min_x, min_y, max_x, max_y = patch.bounds
    pixels_per_world_unit = pixels_per_tile_diameter / tile_diameter
    blur_radius_pixels = blur_radius_diameters * pixels_per_tile_diameter
    pad_world = max(tile_diameter, 3.0 * blur_radius_pixels / pixels_per_world_unit)
    min_x -= pad_world
    min_y -= pad_world
    max_x += pad_world
    max_y += pad_world

    width = max(8, math.ceil((max_x - min_x) * pixels_per_world_unit))
    height = max(8, math.ceil((max_y - min_y) * pixels_per_world_unit))
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)

    def to_pixel(point: Point) -> tuple[float, float]:
        x, y = point
        return ((x - min_x) * pixels_per_world_unit, (max_y - y) * pixels_per_world_unit)

    polygons = patch.geoms if isinstance(patch, MultiPolygon) else (patch,)
    for polygon in polygons:
        exterior = [to_pixel((float(x), float(y))) for x, y in polygon.exterior.coords]
        draw.polygon(exterior, fill=0)
        for interior in polygon.interiors:
            hole = [to_pixel((float(x), float(y))) for x, y in interior.coords]
            draw.polygon(hole, fill=255)

    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius_pixels))
    pixels = blurred.load()

    def interpolated_gray(point: Point) -> float:
        px, py = to_pixel(point)
        px = min(max(px, 0.0), width - 1.0)
        py = min(max(py, 0.0), height - 1.0)
        x0 = math.floor(px)
        y0 = math.floor(py)
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)
        tx = px - x0
        ty = py - y0
        top = pixels[x0, y0] * (1.0 - tx) + pixels[x1, y0] * tx
        bottom = pixels[x0, y1] * (1.0 - tx) + pixels[x1, y1] * tx
        return top * (1.0 - ty) + bottom * ty

    scored = []
    for a, b in segments:
        midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        scored.append((255.0 - interpolated_gray(midpoint), (a, b)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def polygon_for(tile: Tile) -> Polygon:
    return Polygon(tile.points)


def patch_union(state: State):
    return unary_union([polygon_for(tile) for tile in state.tiles])


def state_key(state: State, precision: int) -> tuple[tuple[tuple[float, float], ...], ...]:
    polygons = []
    for tile in state.tiles:
        rounded = tuple(sorted((round(x, precision), round(y, precision)) for x, y in tile.points))
        polygons.append(rounded)
    return tuple(sorted(polygons))


def candidate_tiles(
    base_points: tuple[Point, ...],
    segment: tuple[Point, Point],
    allow_reflection: bool,
    dedupe_precision: int,
    allowed_length_pairs: tuple[tuple[float, float], ...] | None,
    length_tolerance: float,
) -> Iterable[Tile]:
    a, b = segment
    boundary_angle = math.atan2(b[1] - a[1], b[0] - a[0])
    boundary_length = math.dist(a, b)
    seen: set[tuple[tuple[float, float], ...]] = set()

    for reflected in ([False, True] if allow_reflection else [False]):
        source_points = tuple((-x, y) for x, y in base_points) if reflected else base_points
        for p, q in edges(source_points):
            if not length_pair_allowed(boundary_length, math.dist(p, q), allowed_length_pairs, length_tolerance):
                continue
            tile_angle = math.atan2(q[1] - p[1], q[0] - p[0])
            angle = boundary_angle + math.pi - tile_angle
            rotated_p = rotate_point(p, angle)
            rotated_q = rotate_point(q, angle)

            alignments = (
                (a[0] - rotated_q[0], a[1] - rotated_q[1]),
                (b[0] - rotated_p[0], b[1] - rotated_p[1]),
            )
            for dx, dy in alignments:
                points = tuple((rx + dx, ry + dy) for rx, ry in (rotate_point(point, angle) for point in source_points))
                key = tuple(sorted((round(x, dedupe_precision), round(y, dedupe_precision)) for x, y in points))
                if key in seen:
                    continue
                seen.add(key)
                yield Tile(points, reflected)


def valid_candidate(
    candidate: Tile,
    patch,
    area_tolerance: float,
    contact_tolerance: float,
) -> bool:
    polygon = polygon_for(candidate)
    if not polygon.is_valid or polygon.area <= area_tolerance:
        return False
    if polygon.intersection(patch).area > area_tolerance:
        return False
    snapped_boundary = snap(polygon.boundary, patch.boundary, contact_tolerance)
    if snapped_boundary.intersection(patch.boundary).length < contact_tolerance:
        return False
    combined = unary_union([patch, polygon])
    if isinstance(combined, MultiPolygon):
        return False
    return True


def candidate_conflict_indices(
    candidate: Tile,
    state: State,
    patch,
    area_tolerance: float,
    contact_tolerance: float,
) -> tuple[bool, tuple[int, ...]]:
    polygon = polygon_for(candidate)
    if not polygon.is_valid or polygon.area <= area_tolerance:
        return False, ()

    conflicts = []
    for index, tile in enumerate(state.tiles):
        if polygon.intersection(polygon_for(tile)).area > area_tolerance:
            conflicts.append(index)
    if conflicts:
        return False, tuple(conflicts)

    snapped_boundary = snap(polygon.boundary, patch.boundary, contact_tolerance)
    if snapped_boundary.intersection(patch.boundary).length < contact_tolerance:
        return False, ()
    combined = unary_union([patch, polygon])
    if isinstance(combined, MultiPolygon):
        return False, ()
    return True, ()


def boundary_length_after(candidate: Tile, patch) -> float:
    return unary_union([patch, polygon_for(candidate)]).boundary.length


def draw_png(state: State, output: Path) -> None:
    all_points = [point for tile in state.tiles for point in tile.points]
    min_x = min(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_x = max(x for x, _ in all_points)
    max_y = max(y for _, y in all_points)
    pad = 1.0
    world_width = max_x - min_x + 2 * pad
    world_height = max_y - min_y + 2 * pad
    scale = 700 / max(world_width, world_height, 1e-9)
    image_width = max(64, round(world_width * scale))
    image_height = max(64, round(world_height * scale))
    supersample = 2

    def screen(point: Point) -> tuple[int, int]:
        x, y = point
        sx = (x - min_x + pad) * scale * supersample
        sy = (max_y + pad - y) * scale * supersample
        return (round(sx), round(sy))

    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (image_width * supersample, image_height * supersample), "#fbfaf6")
    draw = ImageDraw.Draw(image)
    thickness = 1
    stroke_width = max(1, round(thickness * supersample))

    for tile in state.tiles:
        color = "#8ca8c8" if tile.reflected else "#d9b487"
        screen_points = [screen(point) for point in tile.points]
        draw.polygon(screen_points, fill=color)
        draw.line(screen_points + [screen_points[0]], fill="#17202a", width=stroke_width, joint="curve")

    image = image.resize((image_width, image_height), Image.Resampling.BOX)
    image.save(output)


def dfs_explore(
    base_points: tuple[Point, ...],
    output_dir: Path,
    allow_reflection: bool,
    max_tiles: int,
    max_states: int,
    area_tolerance: float,
    contact_tolerance: float,
    key_precision: int,
    score_mode: str,
    angle_samples: int,
    probe_radius: float,
    bitmap_pixels_per_tile_diameter: float,
    bitmap_blur_radius_diameters: float,
    allowed_length_pairs: tuple[tuple[float, float], ...] | None,
    length_tolerance: float,
    export_every: int,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_output in tuple(output_dir.glob("step_*.png")) + tuple(output_dir.glob("step_*.svg")):
        if old_output.is_file():
            old_output.unlink()
    trace_path = output_dir / "trace.csv"

    initial_tiles = (Tile(base_points),)
    initial = State(initial_tiles, 0, frozenset())
    initial = State(initial.tiles, initial.depth, frozenset({state_key(initial, key_precision)}))
    stack: list[State] = [initial]
    tile_diameter = point_diameter(base_points)
    exported = 0
    expanded = 0

    with trace_path.open("w", newline="", encoding="utf-8") as trace_file:
        trace = csv.writer(trace_file)
        trace.writerow(("step", "tiles", "stack_size", "exported", "backjump_to_tiles"))

        while stack and expanded < max_states:
            state = stack.pop()
            should_export = expanded % export_every == 0
            if should_export:
                draw_png(
                    state,
                    output_dir / f"step_{expanded:04d}_tiles_{len(state.tiles):03d}.png",
                )
                exported += 1

            if len(state.tiles) >= max_tiles:
                trace.writerow((expanded, len(state.tiles), len(stack), int(should_export), ""))
                expanded += 1
                continue

            patch = patch_union(state)
            next_items: list[tuple[float, State]] = []
            conflict_indices: set[int] = set()
            if score_mode == "angle":
                scored_segments = boundary_segments_by_interior_score(patch, angle_samples, probe_radius)
            else:
                scored_segments = boundary_segments_by_bitmap_score(
                    patch,
                    tile_diameter,
                    bitmap_pixels_per_tile_diameter,
                    bitmap_blur_radius_diameters,
                )
            for _, segment in scored_segments[:1]:
                directed_segments = (segment, (segment[1], segment[0]))
                for directed_segment in directed_segments:
                    for candidate in candidate_tiles(
                        base_points,
                        directed_segment,
                        allow_reflection,
                        key_precision,
                        allowed_length_pairs,
                        length_tolerance,
                    ):
                        valid, conflicts = candidate_conflict_indices(
                            candidate,
                            state,
                            patch,
                            area_tolerance,
                            contact_tolerance,
                        )
                        conflict_indices.update(conflicts)
                        if not valid:
                            continue
                        next_tiles = state.tiles + (candidate,)
                        next_state = State(next_tiles, state.depth + 1, state.path_keys)
                        key = state_key(next_state, key_precision)
                        if key in state.path_keys:
                            continue
                        next_state = State(next_tiles, state.depth + 1, state.path_keys | {key})
                        next_items.append((boundary_length_after(candidate, patch), next_state))

                        if len(next_items) >= max_states * 20:
                            break
                    if len(next_items) >= max_states * 20:
                        break
                if len(next_items) >= max_states * 20:
                    break
            next_items.sort(key=lambda item: item[0])
            backjump_to_tiles = ""
            if next_items:
                stack.extend(reversed([next_state for _, next_state in next_items]))
            elif conflict_indices:
                latest_conflict = max(conflict_indices)
                backjump_to_tiles = latest_conflict
                stack = [stack_state for stack_state in stack if len(stack_state.tiles) <= latest_conflict]

            trace.writerow((expanded, len(state.tiles), len(stack), int(should_export), backjump_to_tiles))
            expanded += 1

    return exported, expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore polygon tilings with depth-first boundary growth.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--polygon", type=Path, help="JSON file containing [[x,y], ...] polygon vertices")
    source.add_argument("--preset", choices=("hat", "tile11", "square"), default="hat", help="built-in polygon preset")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/dfs"), help="directory for per-state PNGs")
    parser.add_argument("--allow-reflection", action="store_true", help="also try reflected copies of the tile")
    parser.add_argument("--max-tiles", type=int, default=8, help="maximum tiles in a patch")
    parser.add_argument("--max-states", type=int, default=100, help="maximum DFS states to expand")
    parser.add_argument("--export-every", type=int, default=1, help="export one PNG every N expanded DFS states")
    parser.add_argument("--area-tol", type=float, default=1e-7, help="allowed overlap area tolerance")
    parser.add_argument("--contact-tol", type=float, default=1e-6, help="required boundary contact length")
    parser.add_argument("--key-precision", type=int, default=6, help="rounding precision for duplicate state keys")
    parser.add_argument(
        "--allowed-length-pairs",
        type=parse_allowed_length_pairs,
        help="comma-separated allowed glued edge length pairs, for example 1:1,1:2,2:2,sqrt3:sqrt3",
    )
    parser.add_argument("--length-tol", type=float, default=1e-6, help="tolerance for matching allowed edge lengths")
    parser.add_argument("--score-mode", choices=("bitmap", "angle"), default="bitmap", help="boundary edge score to use")
    parser.add_argument("--angle-samples", type=int, default=72, help="samples around each boundary vertex for filled-angle scoring")
    parser.add_argument("--probe-radius", type=float, default=0.05, help="probe radius for filled-angle scoring")
    parser.add_argument(
        "--bitmap-pixels-per-tile-diameter",
        type=float,
        default=10.0,
        help="bitmap scoring resolution; one tile diameter maps to this many pixels",
    )
    parser.add_argument(
        "--bitmap-blur-radius-diameters",
        type=float,
        default=1.0,
        help="Gaussian blur radius for bitmap scoring, measured in tile diameters",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tiles < 1:
        raise SystemExit("--max-tiles must be at least 1")
    if args.max_states < 1:
        raise SystemExit("--max-states must be at least 1")
    if args.export_every < 1:
        raise SystemExit("--export-every must be at least 1")
    if args.length_tol < 0:
        raise SystemExit("--length-tol must be non-negative")
    if args.bitmap_pixels_per_tile_diameter <= 0:
        raise SystemExit("--bitmap-pixels-per-tile-diameter must be positive")
    if args.bitmap_blur_radius_diameters < 0:
        raise SystemExit("--bitmap-blur-radius-diameters must be non-negative")

    points = load_polygon(args.polygon) if args.polygon else preset_polygon(args.preset)
    base_points = normalize_base(points)
    exported, expanded = dfs_explore(
        base_points,
        args.output_dir,
        args.allow_reflection,
        args.max_tiles,
        args.max_states,
        args.area_tol,
        args.contact_tol,
        args.key_precision,
        args.score_mode,
        args.angle_samples,
        args.probe_radius,
        args.bitmap_pixels_per_tile_diameter,
        args.bitmap_blur_radius_diameters,
        args.allowed_length_pairs,
        args.length_tol,
        args.export_every,
    )
    print(f"Exported {exported} PNGs after expanding {expanded} DFS states into {args.output_dir}.")


if __name__ == "__main__":
    main()
