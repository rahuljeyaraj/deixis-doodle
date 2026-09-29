"""Local speech recognition with word timestamps (sherpa-onnx on CPU).

    asr = LocalAsr()                                   # default model from pointdraw.audio.models
    words = asr.transcribe(samples, 16000, t_start)    # list[deixis.Word] on the shared clock

`t_start` is the `clock.now()` time of the first sample, so word times come out on the same clock
as camera frames and holds. For streaming (cache-aware) models, `LocalAsr.stream(t_start)` decodes
while the person is still speaking, so only the last chunk is left to do when they stop.

Word text keeps the model's casing but not its punctuation ("Here," becomes "Here").
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deixis.types import Word
from pointdraw.audio.models import ASR_MODELS, DEFAULT_ASR, MODELS_DIR, AsrModel
from pointdraw.audio.ort import import_sherpa
from pointdraw.audio.wavio import SAMPLE_RATE, resample_linear

log = logging.getLogger(__name__)

HOTWORDS = ("Nemotron", "Cosmos", "Nebius")
FRAME_S = 0.08  # encoder frame (8x subsampling of 10 ms features) for these NeMo models
TAIL_PAD_S = 0.8  # silence fed after an utterance so a streaming model flushes its last chunk


@dataclass
class AsrConfig:
    model: str = DEFAULT_ASR
    models_dir: Path = MODELS_DIR
    num_threads: int = 4
    hotwords: tuple[str, ...] = HOTWORDS
    hotwords_score: float = 2.0
    use_hotwords: bool = False  # needs beam search; see docs/spikes/r2-asr.md
    # Seconds to subtract from every word time: the model's median emission lag, measured in R2.
    # None means "use the model's measured value" (MEASURED_LAG_S).
    lag_s: float | None = None


# Median lag of demonstrative start times against ground truth (synthetic voices, 300 ms
# pre-roll), from scripts/asr_bench.py on 30 Sep 2026; see docs/spikes/r2-asr.md. Re-check with
# real speech once the clap test (audio/sync.py) has calibrated the audio clock.
MEASURED_LAG_S: dict[str, float] = {
    "parakeet_v2": 0.095,
    "nemotron35": 0.135,
    "nemotron_en": 0.025,
}


def tokens_to_words(
    tokens: list[str],
    timestamps: list[float],
    durations: list[float] | None = None,
    t_end: float | None = None,
) -> list[tuple[str, float, float]]:
    """Group sub-word tokens into (word, t0, t1) with times relative to the audio start.

    sherpa-onnx marks a word start with a leading space (SentencePiece's "▁"). Some models emit
    a bare " " token before the word's first piece. Punctuation-only tokens are dropped.
    """
    words: list[list] = []  # [text, t0, t_last, dur_last]
    new_word = True
    for k, (tok, ts) in enumerate(zip(tokens, timestamps, strict=False)):
        tok = tok.replace("▁", " ")
        dur = durations[k] if durations and k < len(durations) else FRAME_S
        if tok.startswith(" "):
            new_word = True
        piece = tok.strip()
        if not piece:
            continue
        if not any(c.isalnum() for c in piece):
            new_word = True  # punctuation ends a word
            continue
        if new_word or not words:
            words.append([piece, ts, ts, dur])
            new_word = False
        else:
            words[-1][0] += piece
            words[-1][2] = ts
            words[-1][3] = dur
    out: list[tuple[str, float, float]] = []
    for i, (text, t0, t_last, dur) in enumerate(words):
        t1 = t_last + max(dur, FRAME_S)
        if i + 1 < len(words):
            t1 = min(t1, words[i + 1][1])
        elif t_end is not None:
            t1 = min(t1, t_end)
        out.append((text.strip("'"), float(t0), float(max(t1, t0))))
    return [w for w in out if w[0]]


def greedy_tokenize(word: str, vocab: set[str]) -> list[str] | None:
    """Split a word into the model's BPE pieces by greedy longest match (for hotword files)."""
    pieces: list[str] = []
    s = "▁" + word
    while s:
        for n in range(len(s), 0, -1):
            if s[:n] in vocab:
                pieces.append(s[:n])
                s = s[n:]
                break
        else:
            return None
    return pieces


class AsrStream:
    """Incremental decoding of one utterance with a streaming model."""

    def __init__(self, asr: LocalAsr, t_start: float) -> None:
        self._asr = asr
        self._stream = asr._new_online_stream()
        self.t_start = t_start
        self.n_samples = 0

    def accept(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
        """Feed audio as it arrives and decode every chunk that is ready."""
        x = resample_linear(np.asarray(samples, np.float32), sample_rate)
        self.n_samples += len(x)
        self._stream.accept_waveform(SAMPLE_RATE, x)
        rec = self._asr._recognizer
        while rec.is_ready(self._stream):
            rec.decode_stream(self._stream)

    def finish(self) -> list[Word]:
        """Flush the last chunk and return the words on the shared clock."""
        rec = self._asr._recognizer
        self._stream.accept_waveform(
            SAMPLE_RATE, np.zeros(int(TAIL_PAD_S * SAMPLE_RATE), np.float32)
        )
        self._stream.input_finished()
        while rec.is_ready(self._stream):
            rec.decode_stream(self._stream)
        r = rec.get_result_all(self._stream)
        dur = self.n_samples / SAMPLE_RATE
        rel = tokens_to_words(list(r.tokens), list(r.timestamps), None, dur)
        return self._asr._to_words(rel, self.t_start, dur)


class LocalAsr:
    """One loaded sherpa-onnx transducer. Loading takes a few seconds; reuse the instance."""

    def __init__(self, config: AsrConfig | None = None) -> None:
        self.config = config or AsrConfig()
        self.model: AsrModel = ASR_MODELS[self.config.model]
        if not self.model.available(self.config.models_dir):
            raise FileNotFoundError(
                f"{self.model.archive} not found in {self.config.models_dir}; run "
                f"`uv run python scripts/asr_fetch_models.py {self.model.key}`"
            )
        lag = self.config.lag_s
        self.lag_s = MEASURED_LAG_S.get(self.model.key, 0.0) if lag is None else lag
        self._hotwords_file: str = ""
        t = time.perf_counter()
        self._recognizer = self._load()
        self.load_s = time.perf_counter() - t
        log.info("loaded %s in %.1f s", self.model.key, self.load_s)

    @property
    def streaming(self) -> bool:
        return self.model.kind == "online"

    def _write_hotwords(self, tokens_txt: Path) -> str:
        vocab = {line.split(" ")[0] for line in tokens_txt.read_text(encoding="utf-8").splitlines()}
        lines = []
        for w in self.config.hotwords:
            for variant in {w, w.lower()}:
                pieces = greedy_tokenize(variant, vocab)
                if pieces:
                    lines.append(" ".join(pieces) + f" :{self.config.hotwords_score}")
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8", prefix="hotwords_"
        )
        f.write("\n".join(lines) + "\n")
        f.close()
        return f.name

    def _load(self):
        so = import_sherpa()
        files = {k: str(v) for k, v in self.model.files(self.config.models_dir).items()}
        kw: dict = {"num_threads": self.config.num_threads, "decoding_method": "greedy_search"}
        if self.config.use_hotwords and self.streaming:
            # sherpa-onnx 1.13.8 has no beam search for streaming NeMo transducers and exits the
            # whole process if asked (fix merged upstream 14 Sep 2026, PR #3895, not released).
            log.warning("hotwords not supported for streaming %s yet; ignoring", self.model.key)
            self.config.use_hotwords = False
        if self.config.use_hotwords:
            self._hotwords_file = self._write_hotwords(Path(files["tokens"]))
            kw.update(
                decoding_method="modified_beam_search",
                hotwords_file=self._hotwords_file,
                hotwords_score=self.config.hotwords_score,
            )
        if self.streaming:
            return so.OnlineRecognizer.from_transducer(**files, **kw)
        return so.OfflineRecognizer.from_transducer(**files, model_type="nemo_transducer", **kw)

    def _new_online_stream(self):
        s = self._recognizer.create_stream()
        if self.model.language:
            s.set_option("language", self.model.language)
        return s

    def stream(self, t_start: float) -> AsrStream:
        """Start incremental decoding of an utterance whose first sample is at `t_start`."""
        if not self.streaming:
            raise ValueError(f"{self.model.key} is not a streaming model; use transcribe()")
        return AsrStream(self, t_start)

    def _to_words(
        self, rel: list[tuple[str, float, float]], t_start: float, dur: float
    ) -> list[Word]:
        out = []
        for text, t0, t1 in rel:
            a = min(max(t0 - self.lag_s, 0.0), dur)
            b = min(max(t1 - self.lag_s, a), dur)
            out.append(Word(text=text, t0=t_start + a, t1=t_start + b))
        return out

    def transcribe(
        self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE, t_start: float = 0.0
    ) -> list[Word]:
        """Recognise one utterance. Word times are `t_start` + offset into `samples`."""
        x = resample_linear(np.asarray(samples, np.float32).reshape(-1), sample_rate)
        dur = len(x) / SAMPLE_RATE
        if self.streaming:
            s = AsrStream(self, t_start)
            s.accept(x)
            return s.finish()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, x)
        self._recognizer.decode_stream(stream)
        r = stream.result
        durations = list(getattr(r, "durations", []) or [])
        rel = tokens_to_words(list(r.tokens), list(r.timestamps), durations, dur)
        return self._to_words(rel, t_start, dur)
