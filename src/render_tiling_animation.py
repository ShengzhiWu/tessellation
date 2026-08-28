#!/usr/bin/env python3
"""Render a tiling search trace as a PNG sequence plus generated stereo audio."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from tiling_explorer import Point, normalize_base, preset_polygon, rotate_point


STATE_RE = re.compile(r"^state_(\d+)_tiles_(\d+)\.h5$")


@dataclass(frozen=True)
class TileInstance:
    x: float
    y: float
    angle: float
    reflected: bool


@dataclass
class Camera:
    cx: float
    cy: float
    zoom: float


@dataclass(frozen=True)
class FrameState:
    step: int
    tiles: list[TileInstance]
    birth_steps: list[int]


@dataclass(frozen=True)
class RemovalEvent:
    time: float
    center: Point


def parse_state_files(input_dir: Path) -> dict[int, Path]:
    states: dict[int, Path] = {}
    for path in input_dir.glob("state_*_tiles_*.h5"):
        match = STATE_RE.match(path.name)
        if match:
            states[int(match.group(1))] = path
    return states


def load_state(path: Path) -> list[TileInstance]:
    with h5py.File(path, "r") as file:
        transforms = np.asarray(file["transforms"], dtype=np.float64)
        reflected = np.asarray(file["reflected"], dtype=bool)
    return [
        TileInstance(float(x), float(y), float(angle), bool(is_reflected))
        for (x, y, angle), is_reflected in zip(transforms, reflected)
    ]


def transform_points(base_points: tuple[Point, ...], tile: TileInstance) -> list[Point]:
    source = tuple((-x, y) for x, y in base_points) if tile.reflected else base_points
    return [(rx + tile.x, ry + tile.y) for rx, ry in (rotate_point(point, tile.angle) for point in source)]


def tile_centroid(base_points: tuple[Point, ...], tile: TileInstance) -> Point:
    points = transform_points(base_points, tile)
    return (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points))


def tile_key(tile: TileInstance, precision: int = 9) -> tuple[float, float, float, bool]:
    return (round(tile.x, precision), round(tile.y, precision), round(tile.angle, precision), tile.reflected)


def common_prefix_length(first: list[TileInstance], second: list[TileInstance]) -> int:
    limit = min(len(first), len(second))
    for index in range(limit):
        if tile_key(first[index]) != tile_key(second[index]):
            return index
    return limit


def state_bounds(base_points: tuple[Point, ...], state: list[TileInstance]) -> tuple[float, float, float, float]:
    all_points = [point for tile in state for point in transform_points(base_points, tile)]
    min_x = min(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_x = max(x for x, _ in all_points)
    max_y = max(y for _, y in all_points)
    return min_x, min_y, max_x, max_y


def target_camera(
    base_points: tuple[Point, ...],
    state: list[TileInstance],
    width: int,
    height: int,
    fill: float,
) -> Camera:
    min_x, min_y, max_x, max_y = state_bounds(base_points, state)
    world_width = max(max_x - min_x, 1e-9)
    world_height = max(max_y - min_y, 1e-9)
    zoom = fill * min(width / world_width, height / world_height)
    return Camera((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, zoom)


def smooth_camera(current: Camera, target: Camera, alpha: float) -> Camera:
    return Camera(
        (1.0 - alpha) * current.cx + alpha * target.cx,
        (1.0 - alpha) * current.cy + alpha * target.cy,
        (1.0 - alpha) * current.zoom + alpha * target.zoom,
    )


def lerp(first: float, second: float, amount: float) -> float:
    return first * (1.0 - amount) + second * amount


def fade(value: float) -> float:
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def gradient(ix: int, iy: int, seed: int) -> tuple[float, float]:
    value = (ix * 374761393 + iy * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    value ^= value >> 13
    value = (value * 1274126177) & 0xFFFFFFFF
    value ^= value >> 16
    angle = (value / 0x100000000) * 2.0 * math.pi
    return (math.cos(angle), math.sin(angle))


def perlin(x: float, y: float, seed: int = 0) -> float:
    x0 = math.floor(x)
    y0 = math.floor(y)
    xf = x - x0
    yf = y - y0

    def dot_grid(ix: int, iy: int) -> float:
        gx, gy = gradient(ix, iy, seed)
        return gx * (x - ix) + gy * (y - iy)

    u = fade(xf)
    v = fade(yf)
    top = lerp(dot_grid(x0, y0), dot_grid(x0 + 1, y0), u)
    bottom = lerp(dot_grid(x0, y0 + 1), dot_grid(x0 + 1, y0 + 1), u)
    return max(0.0, min(1.0, 0.5 + lerp(top, bottom, v)))


def tile_phase(tile: TileInstance) -> float:
    key = tile_key(tile, precision=5)
    text = f"{key[0]}:{key[1]}:{key[2]}:{int(key[3])}"
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value / 0x100000000


def blend_rgb(first: tuple[int, int, int], second: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(round(lerp(a, b, amount)) for a, b in zip(first, second))


def tile_style(
    base_points: tuple[Point, ...],
    tile: TileInstance,
    birth_step: int,
    current_step: int,
    current_time: float,
    end_step: int,
    duration: float,
    tile_diameter: float,
    noise_spatial_scale: float,
    noise_speed: float,
    fill_alpha: float,
    noise_alpha_amplitude: float,
    pulse_alpha_amplitude: float,
    red_shift: float,
    red_decay_seconds: float,
) -> tuple[tuple[int, int, int], int]:
    base_color = (74, 163, 255) if tile.reflected else (255, 213, 74)
    age_seconds = max(0.0, (current_step - birth_step) / max(end_step, 1) * duration)
    color = blend_rgb(base_color, (255, 48, 48), red_shift * math.exp(-age_seconds / red_decay_seconds))

    centroid_x, centroid_y = tile_centroid(base_points, tile)
    noise_scale = max(noise_spatial_scale * tile_diameter, 1e-9)
    noise_value = perlin(
        centroid_x / noise_scale + current_time * noise_speed,
        centroid_y / noise_scale + current_time * noise_speed * 0.43,
        seed=17,
    )
    pulse = math.sin(2.0 * math.pi * (0.28 * current_time + tile_phase(tile)))
    alpha = fill_alpha + noise_alpha_amplitude * (2.0 * noise_value - 1.0) + pulse_alpha_amplitude * pulse
    return color, round(255 * max(0.05, min(1.0, alpha)))


def render_frame(
    base_points: tuple[Point, ...],
    state: list[TileInstance],
    birth_steps: list[int],
    camera: Camera,
    output: Path,
    width: int,
    height: int,
    supersample: int,
    current_step: int,
    end_step: int,
    duration: float,
    tile_diameter: float,
    noise_spatial_scale: float,
    noise_speed: float,
    fill_alpha: float,
    noise_alpha_amplitude: float,
    pulse_alpha_amplitude: float,
    red_shift: float,
    red_decay_seconds: float,
) -> None:
    image = Image.new("RGBA", (width * supersample, height * supersample), (0, 0, 0, 255))
    fill_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    current_time = current_step / max(end_step, 1) * duration

    def screen(point: Point) -> tuple[int, int]:
        x, y = point
        return (
            round(((x - camera.cx) * camera.zoom + width / 2.0) * supersample),
            round((height / 2.0 - (y - camera.cy) * camera.zoom) * supersample),
        )

    stroke_width = max(1, round(1 * supersample))
    rendered: list[tuple[list[tuple[int, int]], tuple[int, int, int]]] = []
    for tile, birth_step in zip(state, birth_steps):
        color, alpha = tile_style(
            base_points,
            tile,
            birth_step,
            current_step,
            current_time,
            end_step,
            duration,
            tile_diameter,
            noise_spatial_scale,
            noise_speed,
            fill_alpha,
            noise_alpha_amplitude,
            pulse_alpha_amplitude,
            red_shift,
            red_decay_seconds,
        )
        screen_points = [screen(point) for point in transform_points(base_points, tile)]
        fill_draw.polygon(screen_points, fill=(*color, alpha))
        rendered.append((screen_points, color))

    image = Image.alpha_composite(image, fill_layer)
    edge_draw = ImageDraw.Draw(image)
    for screen_points, color in rendered:
        edge_draw.line(screen_points + [screen_points[0]], fill=(*color, 255), width=stroke_width, joint="curve")

    image = image.convert("RGB")
    image = image.resize((width, height), Image.Resampling.BOX)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_removal_crosses(
    events: list[RemovalEvent],
    camera: Camera,
    output: Path,
    width: int,
    height: int,
    supersample: int,
    current_time: float,
    tile_diameter: float,
    half_life: float,
    size_diameters: float,
    stroke_fraction: float,
) -> None:
    image = Image.new("RGBA", (width * supersample, height * supersample), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    visible_lifetime = half_life * 8.0

    def screen(point: Point) -> tuple[int, int]:
        x, y = point
        return (
            round(((x - camera.cx) * camera.zoom + width / 2.0) * supersample),
            round((height / 2.0 - (y - camera.cy) * camera.zoom) * supersample),
        )

    cross_length = max(4, round(tile_diameter * camera.zoom * size_diameters * supersample))
    red_width = max(2, round(cross_length * stroke_fraction))
    black_width = max(red_width + 2, round(red_width * 1.65))

    for event in events:
        age = current_time - event.time
        if age < 0.0 or age > visible_lifetime:
            continue
        alpha = round(255.0 * math.exp(-math.log(2.0) * age / half_life))
        if alpha <= 0:
            continue
        cx, cy = screen(event.center)
        half = cross_length // 2
        first = ((cx - half, cy - half), (cx + half, cy + half))
        second = ((cx - half, cy + half), (cx + half, cy - half))
        draw.line(first, fill=(0, 0, 0, alpha), width=black_width)
        draw.line(second, fill=(0, 0, 0, alpha), width=black_width)
        draw.line(first, fill=(255, 38, 38, alpha), width=red_width)
        draw.line(second, fill=(255, 38, 38, alpha), width=red_width)

    image = image.resize((width, height), Image.Resampling.BOX)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def event_pan(base_points: tuple[Point, ...], state: list[TileInstance], tile: TileInstance) -> float:
    min_x, _, max_x, _ = state_bounds(base_points, state)
    center_x = (min_x + max_x) / 2.0
    half_width = max((max_x - min_x) / 2.0, 1e-9)
    x, _ = tile_centroid(base_points, tile)
    return max(-1.0, min(1.0, (x - center_x) / half_width))


def collect_audio_events(
    base_points: tuple[Point, ...],
    state_files: dict[int, Path],
    start_step: int,
    end_step: int,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260825)
    previous = load_state(state_files[start_step])
    add_events: list[tuple[float, float, float]] = []
    remove_events: list[tuple[float, float, float]] = []

    for step in range(start_step + 1, end_step + 1):
        current = load_state(state_files[step])
        prefix = common_prefix_length(previous, current)
        base_event_time = (step - start_step) / (end_step - start_step) * duration

        for tile in previous[prefix:]:
            remove_events.append((base_event_time, event_pan(base_points, previous, tile), rng.uniform(0.0, 1.0)))
        for tile in current[prefix:]:
            add_events.append((base_event_time, event_pan(base_points, current, tile), rng.uniform(0.0, 1.0)))
        previous = current

    return np.array(add_events, dtype=np.float64).reshape(-1, 3), np.array(remove_events, dtype=np.float64).reshape(-1, 3)


def collect_removal_events(
    base_points: tuple[Point, ...],
    state_files: dict[int, Path],
    start_step: int,
    end_step: int,
    duration: float,
) -> list[RemovalEvent]:
    previous = load_state(state_files[start_step])
    events: list[RemovalEvent] = []

    for step in range(start_step + 1, end_step + 1):
        current = load_state(state_files[step])
        prefix = common_prefix_length(previous, current)
        base_event_time = (step - start_step) / (end_step - start_step) * duration
        for tile in previous[prefix:]:
            events.append(RemovalEvent(base_event_time, tile_centroid(base_points, tile)))
        previous = current

    return events


def save_audio_events(
    output: Path,
    add_events: np.ndarray,
    remove_events: np.ndarray,
    start_step: int,
    end_step: int,
    duration: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as file:
        file.create_dataset("add", data=add_events)
        file.create_dataset("remove", data=remove_events)
        file.attrs["columns"] = "time,pan,gain"
        file.attrs["start_step"] = start_step
        file.attrs["end_step"] = end_step
        file.attrs["duration"] = duration


def sampled_steps(start_step: int, end_step: int, frame_count: int) -> list[int]:
    if frame_count == 1:
        return [start_step]
    return [
        round(start_step + frame_index * (end_step - start_step) / (frame_count - 1))
        for frame_index in range(frame_count)
    ]


def collect_frame_states(
    state_files: dict[int, Path],
    start_step: int,
    end_step: int,
    frame_steps: list[int],
) -> list[FrameState]:
    requested: dict[int, list[int]] = {}
    for frame_index, step in enumerate(frame_steps):
        requested.setdefault(step, []).append(frame_index)

    frames: list[FrameState | None] = [None] * len(frame_steps)
    previous = load_state(state_files[start_step])
    current_birth_steps = [start_step] * len(previous)
    if start_step in requested:
        for frame_index in requested[start_step]:
            frames[frame_index] = FrameState(start_step, list(previous), list(current_birth_steps))

    for step in range(start_step + 1, end_step + 1):
        current = load_state(state_files[step])
        prefix = common_prefix_length(previous, current)
        current_birth_steps = current_birth_steps[:prefix] + [step] * (len(current) - prefix)
        if step in requested:
            for frame_index in requested[step]:
                frames[frame_index] = FrameState(step, list(current), list(current_birth_steps))
        previous = current

    missing_frames = [index for index, frame in enumerate(frames) if frame is None]
    if missing_frames:
        raise RuntimeError(f"Internal error: missing sampled frame {missing_frames[0]}")
    return [frame for frame in frames if frame is not None]


def render_sequence(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"
    cross_dir = output_dir / "removal_crosses"
    state_files = parse_state_files(input_dir)
    missing = [step for step in range(args.start_step, args.end_step + 1) if step not in state_files]
    if missing:
        raise SystemExit(f"Missing HDF5 state files, first missing step: {missing[0]}")

    base_points = normalize_base(preset_polygon(args.preset))
    if args.events_only:
        add_events, remove_events = collect_audio_events(base_points, state_files, args.start_step, args.end_step, args.duration)
        save_audio_events(output_dir / args.events_name, add_events, remove_events, args.start_step, args.end_step, args.duration)
        return

    frame_dir.mkdir(parents=True, exist_ok=True)
    cross_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()
    for old_cross in cross_dir.glob("frame_*.png"):
        old_cross.unlink()

    frame_count = round(args.duration * args.fps)
    frame_steps = sampled_steps(args.start_step, args.end_step, frame_count)
    frame_states = collect_frame_states(state_files, args.start_step, args.end_step, frame_steps)
    removal_events = collect_removal_events(base_points, state_files, args.start_step, args.end_step, args.duration)
    tile_diameter = max(math.dist(a, b) for index, a in enumerate(base_points) for b in base_points[index + 1 :])
    camera: Camera | None = None
    for frame_index, frame_state in enumerate(frame_states):
        current_time = (frame_state.step - args.start_step) / max(args.end_step - args.start_step, 1) * args.duration
        target = target_camera(base_points, frame_state.tiles, args.width, args.height, args.camera_fill)
        camera = (
            Camera(target.cx, target.cy, target.zoom * args.initial_zoom_factor)
            if camera is None
            else smooth_camera(camera, target, args.camera_alpha)
        )
        render_frame(
            base_points,
            frame_state.tiles,
            frame_state.birth_steps,
            camera,
            frame_dir / f"frame_{frame_index}.png",
            args.width,
            args.height,
            args.supersample,
            frame_state.step,
            args.end_step,
            args.duration,
            tile_diameter,
            args.noise_spatial_scale,
            args.noise_speed,
            args.fill_alpha,
            args.noise_alpha_amplitude,
            args.pulse_alpha_amplitude,
            args.red_shift,
            args.red_decay_seconds,
        )
        render_removal_crosses(
            removal_events,
            camera,
            cross_dir / f"frame_{frame_index}.png",
            args.width,
            args.height,
            args.supersample,
            current_time,
            tile_diameter,
            args.cross_half_life,
            args.cross_size_diameters,
            args.cross_stroke_fraction,
        )
        if args.progress_every and (frame_index + 1) % args.progress_every == 0:
            print(f"rendered {frame_index + 1}/{frame_count} frames")

    add_events, remove_events = collect_audio_events(base_points, state_files, args.start_step, args.end_step, args.duration)
    save_audio_events(output_dir / args.events_name, add_events, remove_events, args.start_step, args.end_step, args.duration)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="outputs/dfs_hat", help="directory containing state_*_tiles_*.h5 files")
    parser.add_argument("--output-dir", default="outputs/dfs_hat_animation", help="directory for frames and audio")
    parser.add_argument("--preset", choices=("hat", "tile11", "square"), default="hat")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=19800)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--camera-fill", type=float, default=0.8)
    parser.add_argument("--camera-alpha", type=float, default=0.05)
    parser.add_argument("--initial-zoom-factor", type=float, default=1.0 / 3.0)
    parser.add_argument("--events-name", default="audio_events.h5")
    parser.add_argument("--events-only", action="store_true", help="write only the audio event HDF5 file and leave existing frames untouched")
    parser.add_argument("--fill-alpha", type=float, default=0.60)
    parser.add_argument("--noise-alpha-amplitude", type=float, default=0.08)
    parser.add_argument("--pulse-alpha-amplitude", type=float, default=0.035)
    parser.add_argument("--noise-spatial-scale", type=float, default=5.0, help="noise wavelength in tile diameters")
    parser.add_argument("--noise-speed", type=float, default=0.035)
    parser.add_argument("--red-shift", type=float, default=0.10)
    parser.add_argument("--red-decay-seconds", type=float, default=1.1)
    parser.add_argument("--cross-half-life", type=float, default=0.15)
    parser.add_argument("--cross-size-diameters", type=float, default=0.35)
    parser.add_argument("--cross-stroke-fraction", type=float, default=0.23)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main() -> None:
    render_sequence(build_parser().parse_args())


if __name__ == "__main__":
    main()
