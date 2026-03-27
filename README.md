# Audio2Transcript

Offline, local audio/video to text + subtitle generator built on faster-whisper.  
Converts any media into 16 kHz mono WAV via ffmpeg, transcribes it, and emits both `.txt` and `.srt` files.

---

## Requirements

- Python 3.9+ (virtual environment recommended). **Python 3.11 is strongly recommended** because PyAV (a faster-whisper dependency) does not yet publish wheels for Python 3.14+, causing install failures.
- FFmpeg CLI (needed for media normalization)
  - macOS (Homebrew): `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg` (or your distro’s package manager)
- Python deps: `python -m pip install -r requirements.txt`

---

## Quick start

```bash
# 1. Optional: create and activate a Python 3.11 venv
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt

# 3. Run the transcriber
python3.11 audio2transcript.py /path/to/media_or_folder \
  -o output \
  --model small \
  --device auto \
  --compute-type auto \
  --beam-size 5 \
  --temperature 0.0
```

The command above processes a single file or every media file inside the supplied directory, dropping transcripts/subtitles (and temporary WAVs) into `output/`.

---

## CLI reference

| Flag | Description | Example / Default |
| --- | --- | --- |
| `input_path` | Positional path to a media file or directory. Hidden files are skipped. | `python audio2transcript.py ~/Videos/talk.mp4` |
| `-o, --output-dir` | Destination directory for `.txt`, `.srt`, and temp WAVs. Created if absent. | `--output-dir output` (default `output`) |
| `-m, --model` | faster-whisper checkpoint to load. | Options: `tiny`, `base`, `small`, `medium`, `large-v3` (default `small`) |
| `--device` | Execution device. `auto` tries CUDA first then CPU. | `--device cuda` or `--device cpu` (default `auto`) |
| `--compute-type` | Precision/quantization mode. | Examples: `float16`, `int8`, `auto` (default `auto`) |
| `--language` | Force ISO language code; omit for autodetect. | `--language en` |
| `--beam-size` | Beam search width (higher = potentially better accuracy, slower). | `--beam-size 5` (default `5`) |
| `--temperature` | Sampling temperature (0.0 = deterministic). | `--temperature 0.0` (default `0.0`) |
| `--keep-temp` | Preserve intermediate WAVs instead of deleting. | Add flag when debugging conversions. |

Other behavior:

- Voice-activity detection (`vad_filter=True`) reduces silence and noise segments automatically.
- Temp WAVs live in `<output-dir>/temp_wavs`; they’re deleted after each file unless `--keep-temp` is passed.
- Errors in individual files are logged and the batch run continues for the remaining media.
- If installation fails with PyAV errors under Python 3.14, reinstall using Python 3.11 (or 3.10/3.9) to use the available wheels.

---

## Example workflow

1. Drop assorted `.mp3`, `.m4a`, `.mp4`, etc. files into `~/media`.
2. Run:
   ```bash
   python audio2transcript.py ~/media -o transcripts --model base --device auto --compute-type auto --beam-size 3 --temperature 0.0
   ```
3. Collect `~/media/<name>.txt` and `.srt` output under `transcripts/`.

You now have plain text transcripts and timestamped subtitles ready for editing or publishing—all without any cloud calls.

