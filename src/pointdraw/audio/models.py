"""Registry of the local speech models (sherpa-onnx exports) and where they live on disk.

Models are downloaded by `scripts/asr_fetch_models.py` into `.cache/models/` and never committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SHERPA_ASR_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
MODELS_DIR = Path(".cache") / "models"


@dataclass(frozen=True)
class AsrModel:
    """One sherpa-onnx transducer export."""

    key: str
    archive: str  # release asset name without .tar.bz2
    kind: Literal["online", "offline"]  # streaming (cache-aware) vs whole-utterance
    licence: str
    source: str  # upstream model card
    language: str | None = None  # multilingual models take a language option

    @property
    def url(self) -> str:
        return f"{SHERPA_ASR_RELEASE}/{self.archive}.tar.bz2"

    def dir(self, models_dir: Path = MODELS_DIR) -> Path:
        return Path(models_dir) / self.archive

    def files(self, models_dir: Path = MODELS_DIR) -> dict[str, Path]:
        d = self.dir(models_dir)
        return {
            "encoder": d / "encoder.int8.onnx",
            "decoder": d / "decoder.int8.onnx",
            "joiner": d / "joiner.int8.onnx",
            "tokens": d / "tokens.txt",
        }

    def available(self, models_dir: Path = MODELS_DIR) -> bool:
        return all(p.exists() for p in self.files(models_dir).values())


ASR_MODELS: dict[str, AsrModel] = {
    m.key: m
    for m in [
        AsrModel(
            key="nemotron35",
            archive="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
            kind="online",
            licence="OpenMDW-1.1",
            source="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
            language="en",
        ),
        AsrModel(
            key="nemotron_en",
            archive="sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
            kind="online",
            licence="NVIDIA Open Model License",
            source="https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b",
        ),
        AsrModel(
            key="parakeet_v2",
            archive="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
            kind="offline",
            licence="CC-BY-4.0",
            source="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
        ),
    ]
}

DEFAULT_ASR = "nemotron_en"

SILERO_VAD_URL = f"{SHERPA_ASR_RELEASE}/silero_vad.onnx"
SILERO_VAD_PATH = MODELS_DIR / "silero_vad.onnx"
