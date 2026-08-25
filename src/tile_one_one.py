#!/usr/bin/env python3
"""Visualize finite patches of the weakly chiral Tile(1,1) tiling.

The script implements the substitution described for Tile(1,1), the
straight-edged base tile of the Spectre family.  It draws one finite
supertile patch, using rotations/translations only within a generated patch
up to a global handedness choice.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SQRT3 = math.sqrt(3.0)
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
TILE_NAMES = ("Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi")

COLORS = {
    "Gamma1": "#dfe5c7",
    "Gamma2": "#aeb47a",
    "Delta": "#e8ecef",
    "Theta": "#f7b7a3",
    "Lambda": "#e06b58",
    "Xi": "#f2d84b",
    "Pi": "#79b8d1",
    "Sigma": "#8e7cc3",
    "Phi": "#69b578",
    "Psi": "#5fb7aa",
}


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Tile:
    label: str
    quad: tuple[Point, Point, Point, Point]

    def iter_tiles(self, transform: tuple[float, ...]) -> Iterable[tuple[str, tuple[float, ...]]]:
        yield self.label, transform


@dataclass(frozen=True)
class MetaTile:
    geometries: tuple[tuple[object, tuple[float, ...]], ...]
    quad: tuple[Point, Point, Point, Point]

    def iter_tiles(self, transform: tuple[float, ...]) -> Iterable[tuple[str, tuple[float, ...]]]:
        for shape, child_transform in self.geometries:
            yield from shape.iter_tiles(mul(transform, child_transform))


# Vertices of Tile(1,1), counterclockwise.  One vertex lies on a straight angle,
# so the equilateral polygon has 14 vertices but visually 13 corners.
TILE_POINTS = (
    Point(0.0, 0.0),
    Point(1.0, 0.0),
    Point(1.5, -SQRT3 / 2.0),
    Point(1.5 + SQRT3 / 2.0, 0.5 - SQRT3 / 2.0),
    Point(1.5 + SQRT3 / 2.0, 1.5 - SQRT3 / 2.0),
    Point(2.5 + SQRT3 / 2.0, 1.5 - SQRT3 / 2.0),
    Point(3.0 + SQRT3 / 2.0, 1.5),
    Point(3.0, 2.0),
    Point(3.0 - SQRT3 / 2.0, 1.5),
    Point(2.5 - SQRT3 / 2.0, 1.5 + SQRT3 / 2.0),
    Point(1.5 - SQRT3 / 2.0, 1.5 + SQRT3 / 2.0),
    Point(0.5 - SQRT3 / 2.0, 1.5 + SQRT3 / 2.0),
    Point(-SQRT3 / 2.0, 1.5),
    Point(0.0, 1.0),
)


def mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    """Compose affine transforms stored as (a,b,c,d,e,f)."""
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


def transformed_polygon(matrix: tuple[float, ...]) -> list[tuple[float, float]]:
    return [(p.x, p.y) for p in (transform_point(matrix, point) for point in TILE_POINTS)]


def build_base_tiles() -> dict[str, object]:
    quad = (TILE_POINTS[3], TILE_POINTS[5], TILE_POINTS[7], TILE_POINTS[11])
    tiles: dict[str, object] = {label: Tile(label, quad) for label in TILE_NAMES if label != "Gamma"}

    mystic = MetaTile(
        (
            (Tile("Gamma1", quad), IDENTITY),
            (
                Tile("Gamma2", quad),
                mul(translate(TILE_POINTS[8].x, TILE_POINTS[8].y), rotate(30.0)),
            ),
        ),
        quad,
    )
    tiles["Gamma"] = mystic
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

    transformations = [mul(reflected, transformation) for transformation in transformations]
    return transformations


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


def generate_tiles(
    iterations: int,
    root: str,
    flip: bool,
    seed_orientation: bool = False,
) -> list[tuple[str, tuple[float, ...]]]:
    tile_system = build_base_tiles()
    for _ in range(iterations):
        tile_system = build_supertiles(tile_system)

    transform = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0) if flip else IDENTITY
    if seed_orientation and iterations == 0:
        delta_child_transform = substitution_transforms(tile_system["Delta"].quad)[1]
        # The substitution placement for this child is reflected.  Use the
        # opposite global handedness so n=0 has the same visual convention as
        # the larger Delta supertiles.
        canonical_handedness = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        transform = mul(mul(transform, canonical_handedness), delta_child_transform)
    return list(tile_system[root].iter_tiles(transform))


def draw_tiling(
    tiles: list[tuple[str, tuple[float, ...]]],
    output: Path,
    show_labels: bool,
    dpi: int,
) -> None:
    polygons = []
    all_x = []
    all_y = []

    for label, transform in tiles:
        points = transformed_polygon(transform)
        polygons.append((label, points))
        all_x.extend(x for x, _ in points)
        all_y.extend(y for _, y in points)

    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)
    pad = 1.0
    min_x = min(all_x) - pad
    min_y = min(all_y) - pad
    view_width = width + pad * 2
    view_height = height + pad * 2

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
        handle.write(
            '<g stroke="#17202a" stroke-width="0.45" stroke-linejoin="round" '
            'stroke-linecap="round">\n'
        )

        for label, points in polygons:
            scaled = [
                (
                    (x - min_x) * scale,
                    svg_height - (y - min_y) * scale,
                )
                for x, y in points
            ]
            point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in scaled)
            handle.write(f'<polygon points="{point_text}" fill="{COLORS[label]}"/>\n')

        handle.write("</g>\n")

        if show_labels:
            handle.write('<g font-family="Arial, sans-serif" font-size="8" fill="#17202a">\n')
            for label, points in polygons:
                cx = sum(x for x, _ in points) / len(points)
                cy = sum(y for _, y in points) / len(points)
                sx = (cx - min_x) * scale
                sy = svg_height - (cy - min_y) * scale
                handle.write(
                    f'<text x="{sx:.2f}" y="{sy:.2f}" text-anchor="middle" '
                    f'dominant-baseline="central">{label[:2]}</text>\n'
                )
            handle.write("</g>\n")

        handle.write("</svg>\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a finite Tile(1,1) substitution tiling.")
    parser.add_argument("-n", "--iterations", type=int, default=4, help="substitution depth; 3-5 is a good range")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("tile_one_one.svg"),
        help="output SVG path",
    )
    parser.add_argument(
        "--root",
        choices=TILE_NAMES,
        default="Delta",
        help="which supertile type to draw as the finite patch",
    )
    parser.add_argument("--flip", action="store_true", help="draw the opposite global handedness")
    parser.add_argument(
        "--seed-orientation",
        action="store_true",
        help="for n=0, draw the root tile in the recursive Delta-child orientation",
    )
    parser.add_argument("--labels", action="store_true", help="draw short cluster labels on tiles")
    parser.add_argument("--dpi", type=int, default=220, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")

    tiles = generate_tiles(args.iterations, args.root, args.flip, args.seed_orientation)
    draw_tiling(tiles, args.output, args.labels, args.dpi)
    print(f"Wrote {args.output} with {len(tiles)} Tile(1,1) copies.")


if __name__ == "__main__":
    main()
