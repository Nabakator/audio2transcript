#!/usr/bin/env python3
"""
Local offline audio-to-text transcription tool backed by faster-whisper.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import ffmpeg
from faster_whisper import WhisperModel
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local offline audio/video to transcript utility powered by faster-whisper.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to an audio/video file or a directory containing media files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where transcripts/subtitles (and temp WAVs) will be written.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Size of the faster-whisper model to load.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on. 'auto' selects GPU if available.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="faster-whisper compute type (auto, float16, int8, etc.).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force a language code (e.g., 'en'). Defaults to autodetect.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for beam search decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for deterministic decoding.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate WAV files. By default they are deleted after use.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def gather_media_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files: List[Path] = []
    for file_path in sorted(p for p in input_path.rglob("*") if p.is_file()):
        if not is_hidden(file_path):
            files.append(file_path)
    return files


def safe_temp_wav_path(source: Path, temp_dir: Path) -> Path:
    digest = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:8]
    return temp_dir / f"{source.stem}_{digest}.wav"


def convert_to_wav(source: Path, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = safe_temp_wav_path(source, temp_dir)
    try:
        (
            ffmpeg.input(str(source))
            .output(str(target), ac=1, ar=16000, format="wav", loglevel="error")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        raise RuntimeError(f"ffmpeg failed for {source}: {exc.stderr}") from exc
    return target


def run_transcription(
    model: WhisperModel,
    wav_path: Path,
    beam_size: int,
    temperature: float,
    language: Optional[str],
) -> Tuple[str, List[Tuple[int, float, float, str]]]:
    text_segments: List[Tuple[int, float, float, str]] = []
    collected_text: List[str] = []
    for idx, segment in enumerate(
        model.transcribe(
            str(wav_path),
            beam_size=beam_size,
            temperature=temperature,
            language=language,
            vad_filter=True,
        )[0],
        start=1,
    ):
        text = segment.text.strip()
        if not text:
            continue
        collected_text.append(text)
        text_segments.append((idx, segment.start or 0.0, segment.end or 0.0, text))
    return "\n".join(collected_text).strip(), text_segments


def format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_txt(text: str, destination: Path) -> None:
    destination.write_text(text + "\n", encoding="utf-8")


def write_srt(segments: List[Tuple[int, float, float, str]], destination: Path) -> None:
    lines: List[str] = []
    for idx, start, end, text in segments:
        lines.append(str(idx))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(text)
        lines.append("")
    destination.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def ensure_dependencies() -> None:
    if shutil.which("ffmpeg"):
        return
    raise EnvironmentError("ffmpeg executable not found. Please install ffmpeg and try again.")


def process_file(
    model: WhisperModel,
    media_path: Path,
    output_dir: Path,
    temp_dir: Path,
    args: argparse.Namespace,
) -> None:
    wav_path = convert_to_wav(media_path, temp_dir)
    try:
        txt_content, segments = run_transcription(
            model=model,
            wav_path=wav_path,
            beam_size=args.beam_size,
            temperature=args.temperature,
            language=args.language,
        )
        if not segments:
            print(f"[warn] No speech detected in {media_path}")
            return
        base_name = media_path.stem
        txt_path = output_dir / f"{base_name}.txt"
        srt_path = output_dir / f"{base_name}.srt"
        write_txt(txt_content, txt_path)
        write_srt(segments, srt_path)
    finally:
        if not args.keep_temp and wav_path.exists():
            wav_path.unlink()


def main() -> None:
    args = parse_args()
    ensure_dependencies()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "temp_wavs"
    device = resolve_device(args.device)

    print(f"[info] Using device={device}, compute_type={args.compute_type}, model={args.model}")
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=args.compute_type,
    )

    media_files = gather_media_files(args.input_path)
    if not media_files:
        print("No files found to transcribe.")
        sys.exit(1)

    for media_path in tqdm(media_files, desc="Transcribing files"):
        try:
            process_file(model, media_path, args.output_dir, temp_dir, args)
        except Exception as exc:
            print(f"[error] Failed processing {media_path}: {exc}")

    if not args.keep_temp and temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()


if __name__ == "__main__":
    main()

