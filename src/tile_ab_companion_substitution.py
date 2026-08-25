#!/usr/bin/env python3
"""Visualize finite substitution patches for ordinary Tile(a,b) family members.

By default this draws the hat case, Tile(1,sqrt(3)), using the same
hat/turtle substitution layout as the authors' interactive visualization.
Unlike tile_one_one.py, the ordinary-family Gamma cluster contains the
companion Tile(b,a), which is the combinatorial signal of the non-chiral
hat-family construction.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
TILE_NAMES = ("Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi")

COLORS = {
    "Gamma1": "#cfa47d",
    "Gamma2": "#8ca8c8",
    "Delta": "#e8ecef",
    "Theta": "#d0d796",
    "Lambda": "#b8cdb2",
    "Xi": "#d3b190",
    "Pi": "#dac5a1",
    "Sigma": "#bf927e",
    "Phi": "#e4d5a7",
    "Psi": "#e0df9c",
}


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Shape:
    label: str
    points: tuple[Point, ...]
    quad: tuple[Point, Point, Point, Point]

    def iter_shapes(self, transform: tuple[float, ...]) -> Iterable[tuple[str, tuple[Point, ...], tuple[float, ...]]]:
        yield self.label, self.points, transform


@dataclass(frozen=True)
class MetaTile:
    geometries: tuple[tuple[object, tuple[float, ...]], ...]
    quad: tuple[Point, Point, Point, Point]

    def iter_shapes(self, transform: tuple[float, ...]) -> Iterable[tuple[str, tuple[Point, ...], tuple[float, ...]]]:
        for shape, child_transform in self.geometries:
            yield from shape.iter_shapes(mul(transform, child_transform))


def mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return (
        a[0] * b[0] + a[1] * b[3],
        a[0] * b[1] + a[1] * b[4],
        a[0] * b[2] + a[1] * b[5] + a[2],
        a[3] * b[0] + a[4] * b[3],
        a[3] * b[1] + a[4] * b[4],
        a[3] * b[2] + a[4] * b[5] + a[5],
    )


def rotate(degrees: float) -> tuple[float, ...]:
    radians = math.radians(degrees)
    c = math.cos(radians)
    s = math.sin(radians)
    return (c, -s, 0.0, s, c, 0.0)


def translate(tx: float, ty: float) -> tuple[float, ...]:
    return (1.0, 0.0, tx, 0.0, 1.0, ty)


def transform_point(matrix: tuple[float, ...], point: Point) -> Point:
    return Point(
        matrix[0] * point.x + matrix[1] * point.y + matrix[2],
        matrix[3] * point.x + matrix[4] * point.y + matrix[5],
    )


def translate_point_to_point(source: Point, target: Point) -> tuple[float, ...]:
    return translate(target.x - source.x, target.y - source.y)


def points_from_steps(lengths: tuple[float, ...], directions: tuple[float, ...]) -> tuple[Point, ...]:
    points = [Point(0.0, 0.0)]
    x = 0.0
    y = 0.0
    for length, degrees in zip(lengths[:-1], directions[:-1]):
        radians = math.radians(degrees)
        x += length * math.cos(radians)
        y += length * math.sin(radians)
        points.append(Point(x, y))
    return tuple(points)


def build_hat_points(a: float, b: float) -> tuple[Point, ...]:
    """Build the hat-form representative Tile(a,b)."""
    directions = (0, -60, 30, 90, 0, 60, 150, 210, 120, 180, 180, 240, 330, 270)
    lengths = (a, a, b, b, a, a, b, b, a, a, a, a, b, b)
    return points_from_steps(lengths, directions)


def build_turtle_points(a: float, b: float) -> tuple[Point, ...]:
    """Build the companion turtle-form representative for Tile(a,b)."""
    directions = (30, -30, 60, 120, 30, 90, 180, 240, 150, 210, 210, 270, 0, -60)
    lengths = (b, b, a, a, b, b, a, a, b, b, b, b, a, a)
    return points_from_steps(lengths, directions)


def make_shape(label: str, points: tuple[Point, ...]) -> Shape:
    return Shape(label, points, (points[3], points[5], points[7], points[11]))


def build_base_tiles(a: float, b: float, dominant: str) -> dict[str, object]:
    hat_points = build_hat_points(a, b)
    turtle_points = build_turtle_points(a, b)
    if dominant == "ba":
        dominant_points, companion_points = turtle_points, hat_points
    else:
        dominant_points, companion_points = hat_points, turtle_points

    tiles: dict[str, object] = {
        label: make_shape(label, dominant_points)
        for label in TILE_NAMES
        if label != "Gamma"
    }

    if dominant == "ab":
        companion_transform = translate(dominant_points[8].x, dominant_points[8].y)
    else:
        companion_transform = mul(translate(dominant_points[9].x, dominant_points[9].y), rotate(60.0))

    gamma = MetaTile(
        (
            (make_shape("Gamma1", dominant_points), IDENTITY),
            (make_shape("Gamma2", companion_points), companion_transform),
        ),
        (dominant_points[3], dominant_points[5], dominant_points[7], dominant_points[11]),
    )
    tiles["Gamma"] = gamma
    return tiles


def substitution_transforms(quad: tuple[Point, Point, Point, Point]) -> list[tuple[float, ...]]:
    reflected = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    transformations = [IDENTITY]
    total_angle = 0.0
    rotation = IDENTITY
    transformed_quad = list(quad)
    transformation_rules = (
        (60.0, 3, 1),
        (0.0, 2, 0),
        (60.0, 3, 1),
        (60.0, 3, 1),
        (0.0, 2, 0),
        (60.0, 3, 1),
        (-120.0, 3, 3),
    )

    for angle, source_idx, target_idx in transformation_rules:
        if angle:
            total_angle += angle
            rotation = rotate(total_angle)
            transformed_quad = [transform_point(rotation, point) for point in quad]

        placement = translate_point_to_point(
            transformed_quad[target_idx],
            transform_point(transformations[-1], quad[source_idx]),
        )
        transformations.append(mul(placement, rotation))

    return [mul(reflected, transformation) for transformation in transformations]


def build_supertiles(tile_system: dict[str, object]) -> dict[str, MetaTile]:
    quad = tile_system["Delta"].quad
    transformations = substitution_transforms(quad)
    super_rules = {
        "Gamma": ("Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi", "Gamma"),
        "Delta": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
        "Theta": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
        "Lambda": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
        "Xi": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
        "Pi": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
        "Sigma": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma"),
        "Phi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
        "Psi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    }

    super_quad = (
        transform_point(transformations[6], quad[2]),
        transform_point(transformations[5], quad[1]),
        transform_point(transformations[3], quad[2]),
        transform_point(transformations[0], quad[1]),
    )
    return {
        label: MetaTile(
            tuple(
                (tile_system[substitution], transformation)
                for substitution, transformation in zip(substitutions, transformations)
                if substitution is not None
            ),
            super_quad,
        )
        for label, substitutions in super_rules.items()
    }


def transformed_polygon(points: tuple[Point, ...], matrix: tuple[float, ...]) -> list[tuple[float, float]]:
    return [(p.x, p.y) for p in (transform_point(matrix, point) for point in points)]


def generate_shapes(iterations: int, root: str, a: float, b: float, dominant: str) -> list[tuple[str, tuple[Point, ...], tuple[float, ...]]]:
    tile_system = build_base_tiles(a, b, dominant)
    for _ in range(iterations):
        tile_system = build_supertiles(tile_system)
    return list(tile_system[root].iter_shapes(IDENTITY))


def draw_svg(shapes: list[tuple[str, tuple[Point, ...], tuple[float, ...]]], output: Path, show_labels: bool) -> None:
    polygons = []
    all_x = []
    all_y = []
    for label, points, transform in shapes:
        polygon = transformed_polygon(points, transform)
        polygons.append((label, polygon))
        all_x.extend(x for x, _ in polygon)
        all_y.extend(y for _, y in polygon)

    pad = 1.0
    min_x = min(all_x) - pad
    min_y = min(all_y) - pad
    view_width = max(all_x) - min(all_x) + pad * 2
    view_height = max(all_y) - min(all_y) + pad * 2
    svg_width = 1400
    svg_height = max(700, round(svg_width * view_height / max(view_width, 1e-9)))
    scale = svg_width / view_width

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
            f'viewBox="0 0 {svg_width} {svg_height}">\n'
        )
        handle.write('<rect width="100%" height="100%" fill="#fbfaf6"/>\n')
        handle.write('<g stroke="#17202a" stroke-width="0.45" stroke-linejoin="round" stroke-linecap="round">\n')
        for label, polygon in polygons:
            scaled = [((x - min_x) * scale, svg_height - (y - min_y) * scale) for x, y in polygon]
            point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in scaled)
            handle.write(f'<polygon points="{point_text}" fill="{COLORS[label]}"/>\n')
        handle.write("</g>\n")

        if show_labels:
            handle.write('<g font-family="Arial, sans-serif" font-size="8" fill="#17202a">\n')
            for label, polygon in polygons:
                cx = sum(x for x, _ in polygon) / len(polygon)
                cy = sum(y for _, y in polygon) / len(polygon)
                sx = (cx - min_x) * scale
                sy = svg_height - (cy - min_y) * scale
                handle.write(
                    f'<text x="{sx:.2f}" y="{sy:.2f}" text-anchor="middle" '
                    f'dominant-baseline="central">{label[:2]}</text>\n'
                )
            handle.write("</g>\n")
        handle.write("</svg>\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ordinary Tile(a,b) substitution patches.")
    parser.add_argument("-n", "--iterations", type=int, default=4, help="substitution depth")
    parser.add_argument("--a", type=float, default=1.0, help="first edge-length parameter")
    parser.add_argument("--b", type=float, default=math.sqrt(3.0), help="second edge-length parameter")
    parser.add_argument("--root", choices=TILE_NAMES, default="Delta", help="which supertile type to draw")
    parser.add_argument(
        "--dominant",
        choices=("ab", "ba"),
        default="ab",
        help="draw mostly Tile(a,b) or mostly Tile(b,a); default is hat-dominant",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("tile_ab_hat.svg"), help="output SVG path")
    parser.add_argument("--labels", action="store_true", help="draw short cluster labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")
    if args.a <= 0 or args.b <= 0:
        raise SystemExit("--a and --b must both be positive for ordinary family members")
    if math.isclose(args.a, args.b):
        raise SystemExit("Use tile_one_one.py for the special a=b weakly chiral case")

    shapes = generate_shapes(args.iterations, args.root, args.a, args.b, args.dominant)
    draw_svg(shapes, args.output, args.labels)
    print(f"Wrote {args.output} with {len(shapes)} drawn polygons for Tile({args.a:g},{args.b:g}).")


if __name__ == "__main__":
    main()
