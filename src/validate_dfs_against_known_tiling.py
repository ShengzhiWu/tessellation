#!/usr/bin/env python3
"""Validate DFS pruning against a known finite tiling.

The first supported reference source is the Tile(1,1) substitution tiling in
``src/hat/tile_one_one.py``.  The validation still treats the operation as a
general DFS check: align a known finite tiling to the DFS starting tile, run the
real DFS implementation, and stop if DFS tries to backtrack from a state that is
still a subset of the known tiling and whose selected boundary segment is
internal to that known tiling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import LineString

import tiling_explorer as explorer


Matrix = tuple[float, float, float, float, float, float]


class OracleFailure(RuntimeError):
    def __init__(self, state: explorer.State, segment: tuple[explorer.Point, explorer.Point], info: dict[str, object]) -> None:
        super().__init__("DFS tried to backtrack inside the known tiling")
        self.state = state
        self.segment = segment
        self.info = info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DFS backtracking against a known finite tiling.")
    parser.add_argument("--known-source", choices=("tile11-substitution",), default="tile11-substitution")
    parser.add_argument("--iterations", type=int, default=3, help="substitution depth for the known tiling")
    parser.add_argument("--root", default="Delta", help="root supertile for the known tiling")
    parser.add_argument("--flip", action="store_true", help="use the opposite global handedness for the known tiling")
    parser.add_argument("--seed-orientation", action="store_true", help="pass through to tile_one_one.generate_tiles")
    parser.add_argument("--anchor-index", type=int, default=None, help="known tiling tile index to align with the DFS start")
    parser.add_argument(
        "--reference-radius",
        type=float,
        default=18.0,
        help="keep known tiles whose transformed centroids lie within this radius of the anchor",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/dfs_validation"))
    parser.add_argument("--max-tiles", type=int, default=1000)
    parser.add_argument("--max-states", type=int, default=20000)
    parser.add_argument("--export-every", type=int, default=1000000, help="large by default; validation has its own outputs")
    parser.add_argument("--area-tol", type=float, default=1e-7)
    parser.add_argument("--contact-tol", type=float, default=1e-6)
    parser.add_argument("--key-precision", type=int, default=6)
    parser.add_argument("--length-tol", type=float, default=1e-6)
    parser.add_argument("--allowed-length-pairs", default="1:1,1:2")
    parser.add_argument("--score-mode", choices=("bitmap", "angle"), default="bitmap")
    parser.add_argument("--angle-samples", type=int, default=72)
    parser.add_argument("--probe-radius", type=float, default=0.08)
    parser.add_argument("--bitmap-pixels-per-tile-diameter", type=float, default=10.0)
    parser.add_argument("--bitmap-blur-radius-diameters", type=float, default=1.0)
    return parser.parse_args()


def load_tile_one_one_module():
    module_path = Path(__file__).resolve().parent / "hat" / "tile_one_one.py"
    spec = importlib.util.spec_from_file_location("tile_one_one_reference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def transform_point(matrix: Matrix, point: explorer.Point) -> explorer.Point:
    a, b, c, d, e, f = matrix
    x, y = point
    return (a * x + b * y + c, d * x + e * y + f)


def invert_affine(matrix: Matrix) -> Matrix:
    a, b, c, d, e, f = matrix
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("Singular affine transform")
    inv_a = e / det
    inv_b = -b / det
    inv_d = -d / det
    inv_e = a / det
    inv_c = -(inv_a * c + inv_b * f)
    inv_f = -(inv_d * c + inv_e * f)
    return (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f)


def compose_affine(first: Matrix, second: Matrix) -> Matrix:
    a, b, c, d, e, f = first
    g, h, i, j, k, l = second
    return (
        a * g + b * j,
        a * h + b * k,
        a * i + b * l + c,
        d * g + e * j,
        d * h + e * k,
        d * i + e * l + f,
    )


def translate_matrix(dx: float, dy: float) -> Matrix:
    return (1.0, 0.0, dx, 0.0, 1.0, dy)


def matrix_to_tile(matrix: Matrix, base_points: tuple[explorer.Point, ...]) -> explorer.Tile:
    det = matrix[0] * matrix[4] - matrix[1] * matrix[3]
    angle = math.atan2(matrix[3], matrix[0])
    points = tuple(transform_point(matrix, point) for point in base_points)
    return explorer.Tile(points, reflected=det < 0.0, x=matrix[2], y=matrix[5], angle=angle)


def tile_key(tile: explorer.Tile, precision: int) -> tuple[tuple[float, float], ...]:
    return tuple(sorted((round(x, precision), round(y, precision)) for x, y in tile.points))


def json_point(point: explorer.Point) -> list[float]:
    return [point[0], point[1]]


def json_segment(segment: tuple[explorer.Point, explorer.Point]) -> list[list[float]]:
    return [json_point(segment[0]), json_point(segment[1])]


def choose_anchor(matrices: list[Matrix], base_points: tuple[explorer.Point, ...], explicit_index: int | None) -> int:
    if explicit_index is not None:
        if explicit_index < 0 or explicit_index >= len(matrices):
            raise SystemExit("--anchor-index is out of range")
        return explicit_index

    centroids = []
    for matrix in matrices:
        points = [transform_point(matrix, point) for point in base_points]
        centroids.append((sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)))
    center = (sum(x for x, _ in centroids) / len(centroids), sum(y for _, y in centroids) / len(centroids))
    return min(range(len(centroids)), key=lambda index: math.dist(centroids[index], center))


def normalized_tile_matrix(raw_matrix: Matrix, raw_centroid: explorer.Point) -> Matrix:
    return compose_affine(raw_matrix, translate_matrix(raw_centroid[0], raw_centroid[1]))


def build_reference_tiles(
    args: argparse.Namespace,
    base_points: tuple[explorer.Point, ...],
    raw_centroid: explorer.Point,
) -> tuple[tuple[explorer.Tile, ...], int]:
    if args.known_source != "tile11-substitution":
        raise SystemExit(f"Unsupported known source: {args.known_source}")

    module = load_tile_one_one_module()
    generated = module.generate_tiles(args.iterations, args.root, args.flip, args.seed_orientation)
    matrices = [normalized_tile_matrix(tuple(matrix), raw_centroid) for _, matrix in generated]
    anchor_index = choose_anchor(matrices, base_points, args.anchor_index)
    inverse_anchor = invert_affine(matrices[anchor_index])

    reference_tiles = []
    for matrix in matrices:
        aligned_matrix = compose_affine(inverse_anchor, matrix)
        tile = matrix_to_tile(aligned_matrix, base_points)
        cx = sum(x for x, _ in tile.points) / len(tile.points)
        cy = sum(y for _, y in tile.points) / len(tile.points)
        if math.hypot(cx, cy) <= args.reference_radius:
            reference_tiles.append(tile)

    return tuple(reference_tiles), anchor_index


def segment_inside_reference(
    segment: tuple[explorer.Point, explorer.Point],
    reference_state: explorer.State,
    tolerance: float,
) -> bool:
    reference = explorer.patch_union(reference_state)
    boundary_line = LineString(segment)
    if reference.boundary.intersection(boundary_line).length > tolerance:
        return False
    midpoint = ((segment[0][0] + segment[1][0]) / 2.0, (segment[0][1] + segment[1][1]) / 2.0)
    return reference.covers(LineString([segment[0], midpoint])) and reference.covers(LineString([midpoint, segment[1]]))


def reference_frontier_contact_length(
    state: explorer.State,
    segment: tuple[explorer.Point, explorer.Point],
    reference_tiles: tuple[explorer.Tile, ...],
    key_precision: int,
    tolerance: float,
) -> float:
    state_keys = {tile_key(tile, key_precision) for tile in state.tiles}
    segment_line = LineString(segment)
    total = 0.0
    for tile in reference_tiles:
        if tile_key(tile, key_precision) in state_keys:
            continue
        boundary = explorer.polygon_for(tile).boundary
        snapped_segment = explorer.snap(segment_line, boundary, tolerance)
        total += snapped_segment.intersection(boundary).length
    return total


def find_reference_neighbor(
    state: explorer.State,
    segment: tuple[explorer.Point, explorer.Point],
    reference_tiles: tuple[explorer.Tile, ...],
    key_precision: int,
    tolerance: float,
) -> tuple[int | None, float]:
    state_keys = {tile_key(tile, key_precision) for tile in state.tiles}
    segment_line = LineString(segment)
    best_index = None
    best_length = 0.0
    for index, tile in enumerate(reference_tiles):
        if tile_key(tile, key_precision) in state_keys:
            continue
        length = explorer.polygon_for(tile).boundary.intersection(segment_line).length
        if length > best_length:
            best_index = index
            best_length = length
    if best_length <= tolerance:
        patch = explorer.patch_union(state)
        a, b = segment
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= tolerance:
            return None, best_length
        midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        normal = (-dy / length, dx / length)
        probe_distance = max(tolerance * 10.0, 1e-5)
        probe_points = (
            (midpoint[0] + normal[0] * probe_distance, midpoint[1] + normal[1] * probe_distance),
            (midpoint[0] - normal[0] * probe_distance, midpoint[1] - normal[1] * probe_distance),
        )
        outside_probe = None
        for point in probe_points:
            if not patch.covers(explorer.ShapelyPoint(point)):
                outside_probe = explorer.ShapelyPoint(point)
                break
        if outside_probe is None:
            return None, best_length
        for index, tile in enumerate(reference_tiles):
            if tile_key(tile, key_precision) in state_keys:
                continue
            if explorer.polygon_for(tile).covers(outside_probe):
                length = explorer.polygon_for(tile).boundary.intersection(segment_line).length
                return index, length
        return None, best_length
    return best_index, best_length


def find_reference_neighbors_along_segment(
    state: explorer.State,
    segment: tuple[explorer.Point, explorer.Point],
    reference_tiles: tuple[explorer.Tile, ...],
    key_precision: int,
    tolerance: float,
    sample_count: int = 9,
) -> list[dict[str, object]]:
    state_keys = {tile_key(tile, key_precision) for tile in state.tiles}
    patch = explorer.patch_union(state)
    a, b = segment
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length <= tolerance:
        return []

    normal = (-dy / length, dx / length)
    probe_distance = max(tolerance * 10.0, 1e-5)
    found: dict[int, dict[str, object]] = {}
    segment_line = LineString(segment)
    for sample_index in range(1, sample_count + 1):
        t = sample_index / (sample_count + 1)
        point = (a[0] + dx * t, a[1] + dy * t)
        probes = (
            (point[0] + normal[0] * probe_distance, point[1] + normal[1] * probe_distance),
            (point[0] - normal[0] * probe_distance, point[1] - normal[1] * probe_distance),
        )
        outside_probe = None
        for probe in probes:
            if not patch.covers(explorer.ShapelyPoint(probe)):
                outside_probe = explorer.ShapelyPoint(probe)
                break
        if outside_probe is None:
            continue
        for index, tile in enumerate(reference_tiles):
            if tile_key(tile, key_precision) in state_keys:
                continue
            polygon = explorer.polygon_for(tile)
            if not polygon.covers(outside_probe):
                continue
            if index not in found:
                found[index] = {
                    "index": index,
                    "sample_parameters": [],
                    "shared_boundary_length": polygon.boundary.intersection(segment_line).length,
                }
            found[index]["sample_parameters"].append(t)
            break
    return list(found.values())


def candidate_rejection_reason(
    candidate: explorer.Tile,
    state: explorer.State,
    patch,
    area_tolerance: float,
    contact_tolerance: float,
    key_precision: int,
) -> tuple[str, dict[str, object]]:
    polygon = explorer.polygon_for(candidate)
    details: dict[str, object] = {}
    if not polygon.is_valid:
        return "invalid_polygon", details
    if polygon.area <= area_tolerance:
        details["area"] = polygon.area
        return "nonpositive_area", details

    conflicts = []
    for index, tile in enumerate(state.tiles):
        overlap = polygon.intersection(explorer.polygon_for(tile)).area
        if overlap > area_tolerance:
            conflicts.append({"tile": index, "overlap_area": overlap})
    if conflicts:
        details["conflicts"] = conflicts
        return "overlap", details

    snapped_boundary = explorer.snap(polygon.boundary, patch.boundary, contact_tolerance)
    contact_length = snapped_boundary.intersection(patch.boundary).length
    details["contact_length"] = contact_length
    if contact_length < contact_tolerance:
        return "insufficient_contact", details

    combined = explorer.unary_union([patch, polygon])
    if isinstance(combined, explorer.MultiPolygon):
        return "multipolygon", details

    next_state = explorer.State(state.tiles + (candidate,), state.path_keys)
    key = explorer.state_key(next_state, key_precision)
    if key in state.path_keys:
        return "path_key", details

    return "accepted", details


def build_candidate_diagnostics(
    state: explorer.State,
    segment: tuple[explorer.Point, explorer.Point],
    reference_tiles: tuple[explorer.Tile, ...],
    base_points: tuple[explorer.Point, ...],
    args: argparse.Namespace,
) -> dict[str, object]:
    patch = explorer.patch_union(state)
    allowed_pairs = explorer.parse_allowed_length_pairs(args.allowed_length_pairs)
    neighbor_index, shared_length = find_reference_neighbor(
        state,
        segment,
        reference_tiles,
        args.key_precision,
        args.contact_tol,
    )
    reference_neighbor = reference_tiles[neighbor_index] if neighbor_index is not None else None
    reference_key = tile_key(reference_neighbor, args.key_precision) if reference_neighbor is not None else None
    reference_polygon = explorer.polygon_for(reference_neighbor) if reference_neighbor is not None else None

    raw_candidates = []
    raw_candidate_tiles = []
    accepted_count = 0
    matched_raw = None
    matched_accepted = None
    closest = []

    for directed_index, directed_segment in enumerate((segment, (segment[1], segment[0]))):
        for candidate, edge_index, alignment_index, boundary_length, source_length in explorer.candidate_tiles_with_metadata(
            base_points,
            directed_segment,
            False,
            args.key_precision,
            allowed_pairs,
            args.length_tol,
        ):
            key = tile_key(candidate, args.key_precision)
            reason, details = candidate_rejection_reason(
                candidate,
                state,
                patch,
                args.area_tol,
                args.contact_tol,
                args.key_precision,
            )
            if reason == "accepted":
                accepted_count += 1

            distance_to_reference = None
            if reference_polygon is not None:
                distance_to_reference = explorer.polygon_for(candidate).hausdorff_distance(reference_polygon)
                closest.append(
                    {
                        "distance": distance_to_reference,
                        "directed_segment": directed_index,
                        "source_edge": edge_index,
                        "alignment": alignment_index,
                        "reflected": candidate.reflected,
                        "reason": reason,
                        "x": candidate.x,
                        "y": candidate.y,
                        "angle": candidate.angle,
                    }
                )

            record = {
                "directed_segment": directed_index,
                "source_edge": edge_index,
                "alignment": alignment_index,
                "reflected": candidate.reflected,
                "boundary_length": boundary_length,
                "source_edge_length": source_length,
                "x": candidate.x,
                "y": candidate.y,
                "angle": candidate.angle,
                "reason": reason,
                "details": details,
                "distance_to_reference_neighbor": distance_to_reference,
            }
            raw_candidates.append(record)
            raw_candidate_tiles.append((candidate, record, reason))
            if reference_key is not None and key == reference_key:
                matched_raw = record
                if reason == "accepted":
                    matched_accepted = record

    neighbors_along_segment = find_reference_neighbors_along_segment(
        state,
        segment,
        reference_tiles,
        args.key_precision,
        args.contact_tol,
    )
    for neighbor in neighbors_along_segment:
        neighbor_tile = reference_tiles[int(neighbor["index"])]
        neighbor_key = tile_key(neighbor_tile, args.key_precision)
        neighbor_polygon = explorer.polygon_for(neighbor_tile)
        raw_match = None
        accepted_match = None
        closest_distance = None
        closest_record = None
        for candidate, record, reason in raw_candidate_tiles:
            distance = explorer.polygon_for(candidate).hausdorff_distance(neighbor_polygon)
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest_record = record
            if tile_key(candidate, args.key_precision) == neighbor_key:
                raw_match = record
                if reason == "accepted":
                    accepted_match = record
        neighbor["found_in_raw_candidates"] = raw_match is not None
        neighbor["found_in_accepted_candidates"] = accepted_match is not None
        neighbor["matched_raw_candidate"] = raw_match
        neighbor["matched_accepted_candidate"] = accepted_match
        neighbor["closest_candidate_distance"] = closest_distance
        neighbor["closest_candidate"] = closest_record

    closest.sort(key=lambda item: item["distance"])
    return {
        "selected_segment": json_segment(segment),
        "selected_segment_length": math.dist(segment[0], segment[1]),
        "reference_neighbor_index": neighbor_index,
        "reference_neighbor_shared_length": shared_length,
        "reference_neighbors_along_segment": neighbors_along_segment,
        "reference_neighbor_found_in_raw_candidates": matched_raw is not None,
        "reference_neighbor_found_in_accepted_candidates": matched_accepted is not None,
        "matched_raw_candidate": matched_raw,
        "matched_accepted_candidate": matched_accepted,
        "raw_candidate_count": len(raw_candidates),
        "accepted_candidate_count": accepted_count,
        "closest_candidates_to_reference_neighbor": closest[:10],
        "candidates": raw_candidates,
    }


def draw_overlay(
    state: explorer.State,
    reference_tiles: tuple[explorer.Tile, ...],
    segment: tuple[explorer.Point, explorer.Point],
    output: Path,
) -> None:
    all_points = [point for tile in reference_tiles for point in tile.points] + [point for tile in state.tiles for point in tile.points]
    min_x = min(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_x = max(x for x, _ in all_points)
    max_y = max(y for _, y in all_points)
    pad = 1.0
    width = 1200
    height = 900
    scale = min(width / max(max_x - min_x + 2 * pad, 1e-9), height / max(max_y - min_y + 2 * pad, 1e-9))

    def screen(point: explorer.Point) -> tuple[int, int]:
        x, y = point
        return (round((x - min_x + pad) * scale), round(height - (y - min_y + pad) * scale))

    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), "#fbfaf6")
    draw = ImageDraw.Draw(image, "RGBA")
    for tile in reference_tiles:
        points = [screen(point) for point in tile.points]
        draw.polygon(points, fill=(210, 210, 210, 80), outline=(120, 120, 120, 120))
    for tile in state.tiles:
        points = [screen(point) for point in tile.points]
        draw.polygon(points, fill=(217, 180, 135, 210), outline=(20, 30, 42, 255))
    draw.line([screen(segment[0]), screen(segment[1])], fill=(230, 40, 30, 255), width=5)
    image.save(output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_points = explorer.remove_collinear_vertices(explorer.preset_polygon("tile11"))
    raw_polygon = explorer.Polygon(raw_points)
    raw_centroid = (raw_polygon.centroid.x, raw_polygon.centroid.y)
    base_points = explorer.normalize_base(raw_points)
    reference_tiles, anchor_index = build_reference_tiles(args, base_points, raw_centroid)
    if not reference_tiles:
        raise SystemExit("Reference crop is empty; increase --reference-radius.")

    reference_state = explorer.State(reference_tiles, frozenset())
    reference_keys = {tile_key(tile, args.key_precision) for tile in reference_tiles}
    failure_path = args.output_dir / "oracle_failure_state.h5"
    reference_path = args.output_dir / "oracle_reference.h5"

    def on_backtrack(
        state: explorer.State,
        segment: tuple[explorer.Point, explorer.Point] | None,
        conflict_indices: set[int],
        backjump_to_tiles: int | None,
    ) -> None:
        if segment is None:
            return
        state_keys = [tile_key(tile, args.key_precision) for tile in state.tiles]
        if any(key not in reference_keys for key in state_keys):
            return
        if not segment_inside_reference(segment, reference_state, args.contact_tol):
            return
        frontier_contact_length = reference_frontier_contact_length(
            state,
            segment,
            reference_tiles,
            args.key_precision,
            args.contact_tol,
        )
        if frontier_contact_length + args.contact_tol < math.dist(segment[0], segment[1]):
            return
        info = {
            "message": "DFS tried to backtrack while the current state is a subset of the known tiling.",
            "tiles": len(state.tiles),
            "selected_segment": json_segment(segment),
            "reference_frontier_contact_length": frontier_contact_length,
            "conflict_indices": sorted(conflict_indices),
            "backjump_to_tiles": backjump_to_tiles,
        }
        raise OracleFailure(state, segment, info)

    try:
        exported, expanded = explorer.dfs_explore(
            base_points=base_points,
            output_dir=args.output_dir / "dfs_run",
            allow_reflection=False,
            max_tiles=args.max_tiles,
            max_states=args.max_states,
            area_tolerance=args.area_tol,
            contact_tolerance=args.contact_tol,
            key_precision=args.key_precision,
            score_mode=args.score_mode,
            angle_samples=args.angle_samples,
            probe_radius=args.probe_radius,
            bitmap_pixels_per_tile_diameter=args.bitmap_pixels_per_tile_diameter,
            bitmap_blur_radius_diameters=args.bitmap_blur_radius_diameters,
            allowed_length_pairs=explorer.parse_allowed_length_pairs(args.allowed_length_pairs),
            length_tolerance=args.length_tol,
            export_every=args.export_every,
            save_state_h5_files=False,
            on_backtrack=on_backtrack,
        )
    except OracleFailure as failure:
        explorer.save_state_h5(failure.state, failure_path)
        explorer.save_state_h5(reference_state, reference_path)
        draw_overlay(failure.state, reference_tiles, failure.segment, args.output_dir / "oracle_failure_overlay.png")
        (args.output_dir / "oracle_failure.json").write_text(json.dumps(failure.info, indent=2), encoding="utf-8")
        diagnostics = build_candidate_diagnostics(failure.state, failure.segment, reference_tiles, base_points, args)
        (args.output_dir / "candidate_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(f"Found a likely false backtrack after {len(failure.state.tiles)} tiles.")
        print(f"Wrote {failure_path}")
        print(f"Wrote {reference_path}")
        print(f"Wrote {args.output_dir / 'oracle_failure_overlay.png'}")
        print(f"Wrote {args.output_dir / 'oracle_failure.json'}")
        print(f"Wrote {args.output_dir / 'candidate_diagnostics.json'}")
        raise SystemExit(2) from None

    print(
        f"No false backtrack found. DFS expanded {expanded} states, exported {exported} PNGs. "
        f"Reference crop has {len(reference_tiles)} tiles; anchor index {anchor_index}."
    )


if __name__ == "__main__":
    main()
