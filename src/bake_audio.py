#!/usr/bin/env python3
"""Bake tiling animation audio stems from a cached event HDF5 file."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import h5py
import numpy as np


def click_sound(sample_rate: int) -> np.ndarray:
    length = int(0.055 * sample_rate)
    t = np.arange(length, dtype=np.float64) / sample_rate
    envelope = np.exp(-t * 90.0)
    tone = np.sin(2.0 * math.pi * 1800.0 * t) + 0.45 * np.sin(2.0 * math.pi * 3100.0 * t)
    snap = np.zeros(length, dtype=np.float64)
    snap[: max(1, int(0.004 * sample_rate))] = 1.0
    snap *= np.linspace(1.0, 0.0, len(snap), dtype=np.float64)
    return 0.25 * tone * envelope + 0.55 * snap


def puff_sound(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    length = int(0.18 * sample_rate)
    noise = rng.normal(0.0, 1.0, length)
    kernel_size = max(3, int(0.010 * sample_rate))
    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    low_noise = np.convolve(noise, kernel, mode="same")
    t = np.arange(length, dtype=np.float64) / sample_rate
    envelope = np.exp(-t * 16.0) * (1.0 - np.exp(-t * 80.0))
    body = np.sin(2.0 * math.pi * 95.0 * t) * np.exp(-t * 22.0)
    return 0.30 * low_noise * envelope + 0.18 * body


def add_stereo_event(
    audio: np.ndarray,
    mono: np.ndarray,
    event_time: float,
    pan: float,
    sample_rate: int,
    max_delay_seconds: float,
    gain: float = 1.0,
) -> None:
    pan = max(-1.0, min(1.0, pan))
    left_gain = math.sqrt((1.0 - pan) / 2.0)
    right_gain = math.sqrt((1.0 + pan) / 2.0)
    left_delay = max(0, round(max_delay_seconds * sample_rate * pan))
    right_delay = max(0, round(-max_delay_seconds * sample_rate * pan))
    start = round(event_time * sample_rate)

    for channel, channel_gain, delay in ((0, left_gain, left_delay), (1, right_gain, right_delay)):
        channel_start = start + delay
        if channel_start >= len(audio):
            continue
        end = min(len(audio), channel_start + len(mono))
        audio[channel_start:end, channel] += gain * channel_gain * mono[: end - channel_start]


def normalized_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    actual_peak = float(np.max(np.abs(audio)))
    if actual_peak == 0:
        return audio
    return audio * (target_peak / actual_peak)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(pcm.tobytes())


def load_events(path: Path) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    with h5py.File(path, "r") as file:
        add_events = np.asarray(file["add"], dtype=np.float64)
        remove_events = np.asarray(file["remove"], dtype=np.float64)
        duration = float(file.attrs["duration"])
        start_step = int(file.attrs["start_step"])
        end_step = int(file.attrs["end_step"])
    return add_events, remove_events, duration, start_step, end_step


def bake_track(
    events: np.ndarray,
    mono: np.ndarray,
    duration: float,
    sample_rate: int,
    step_seconds: float,
    rng: np.random.Generator,
    jitter_steps: float,
    max_delay_seconds: float,
) -> np.ndarray:
    audio = np.zeros((round(duration * sample_rate), 2), dtype=np.float64)
    for event_time, pan, gain in events:
        jittered_time = event_time + rng.uniform(-jitter_steps, jitter_steps) * step_seconds
        add_stereo_event(
            audio,
            mono,
            max(0.0, min(duration, float(jittered_time))),
            float(pan),
            sample_rate,
            max_delay_seconds=max_delay_seconds,
            gain=float(gain),
        )
    return normalized_audio(audio)


def bake_audio(args: argparse.Namespace) -> None:
    add_events, remove_events, duration, start_step, end_step = load_events(Path(args.events))
    step_seconds = duration / max(end_step - start_step, 1)
    rng = np.random.default_rng(args.seed)
    click = click_sound(args.sample_rate)
    puff = puff_sound(args.sample_rate, rng)
    output_dir = Path(args.output_dir)

    add_audio = bake_track(
        add_events,
        click,
        duration,
        args.sample_rate,
        step_seconds,
        rng,
        args.jitter_steps,
        args.max_delay_seconds,
    )
    remove_audio = bake_track(
        remove_events,
        puff,
        duration,
        args.sample_rate,
        step_seconds,
        rng,
        args.jitter_steps,
        args.max_delay_seconds,
    )
    save_wav(output_dir / args.add_audio_name, add_audio, args.sample_rate)
    save_wav(output_dir / args.remove_audio_name, remove_audio, args.sample_rate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="outputs/dfs_hat_animation/audio_events.h5")
    parser.add_argument("--output-dir", default="outputs/dfs_hat_animation")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--add-audio-name", default="add_sound.wav")
    parser.add_argument("--remove-audio-name", default="remove_sound.wav")
    parser.add_argument("--jitter-steps", type=float, default=0.5)
    parser.add_argument("--max-delay-seconds", type=float, default=0.00065)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main() -> None:
    bake_audio(build_parser().parse_args())


if __name__ == "__main__":
    main()
