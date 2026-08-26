#!/usr/bin/env python3
"""Export the generated add-tile click sound as a standalone WAV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bake_audio import add_stereo_event, click_sound, save_wav


def export_add_tile_sound(output: Path, sample_rate: int, pan: float) -> None:
    mono = click_sound(sample_rate)
    duration = len(mono) / sample_rate + 0.05
    audio = np.zeros((round(duration * sample_rate), 2), dtype=np.float64)
    add_stereo_event(audio, mono, 0.01, pan, sample_rate, max_delay_seconds=0.00065)
    save_wav(output, audio, sample_rate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/add_tile_sound.wav"))
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--pan", type=float, default=0.0, help="-1 is left, 0 is center, 1 is right")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_add_tile_sound(args.output, args.sample_rate, args.pan)


if __name__ == "__main__":
    main()
