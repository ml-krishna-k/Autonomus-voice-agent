#!/usr/bin/env python
"""Download the ASR and TTS models into the image / local cache.

Run at Docker build time so that a cold container never pays a multi-hundred-MB
download on its first request. Also usable locally:

    python scripts/fetch_models.py            # both
    python scripts/fetch_models.py --tts-only
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# voice name -> path fragment inside the rhasspy/piper-voices repo
VOICE_PATHS = {
    "en_US-amy-low": "en/en_US/amy/low",
    "en_US-amy-medium": "en/en_US/amy/medium",
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_US-ryan-high": "en/en_US/ryan/high",
    "en_GB-alba-medium": "en/en_GB/alba/medium",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  = {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  ↓ {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)
    print(f"  ✓ {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def fetch_tts(voice: str, models_dir: Path) -> None:
    print(f"[tts] voice={voice}")
    fragment = VOICE_PATHS.get(voice)
    if fragment is None:
        raise SystemExit(
            f"Unknown voice {voice!r}. Known: {', '.join(sorted(VOICE_PATHS))}.\n"
            "Add it to VOICE_PATHS, or place the .onnx/.onnx.json in models/ manually."
        )
    for suffix in (".onnx", ".onnx.json"):
        _download(
            f"{PIPER_VOICES_BASE}/{fragment}/{voice}{suffix}",
            models_dir / f"{voice}{suffix}",
        )


def fetch_asr(model: str, compute_type: str) -> None:
    print(f"[asr] model={model} compute_type={compute_type}")
    from faster_whisper import WhisperModel

    # Instantiating populates the HuggingFace cache (HF_HOME). We throw the
    # object away; only the download side effect matters here.
    WhisperModel(model, device="cpu", compute_type=compute_type)
    print("  ✓ ASR weights cached")


def main() -> int:
    from app.config import get_settings

    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=settings.tts_voice)
    parser.add_argument("--asr-model", default=settings.asr_model)
    parser.add_argument("--compute-type", default=settings.asr_compute_type)
    parser.add_argument("--models-dir", default=str(settings.models_dir))
    parser.add_argument("--tts-only", action="store_true")
    parser.add_argument("--asr-only", action="store_true")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not args.asr_only:
        fetch_tts(args.voice, models_dir)
    if not args.tts_only:
        fetch_asr(args.asr_model, args.compute_type)

    print("\nAll models ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
