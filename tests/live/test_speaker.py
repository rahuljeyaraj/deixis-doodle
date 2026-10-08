"""The app speaks its replies; the microphone is muted meanwhile (night 2 Oct 2026, item 11)."""

from __future__ import annotations

import time

import numpy as np

from pointdraw.live.speaker import Speaker
from pointdraw.live.speech import Listener


def tone(text: str) -> np.ndarray:
    return np.zeros(int(0.2 * 16000), np.float32)  # 0.2 s of "speech" per reply


def test_mute_covers_each_reply_plus_0_3_s_and_writes_wavs(tmp_path):
    sp = Speaker(tone, play=False, wav_dir=tmp_path, after_s=0.3)
    assert not sp.muted()
    sp.say("Where should it go?")
    sp.say("Stopped.")
    samples: list[tuple[float, bool]] = []
    t_end = time.perf_counter() + 2.0
    while time.perf_counter() < t_end:
        samples.append((time.perf_counter(), sp.muted()))
        time.sleep(0.005)
    sp.close()
    assert [t for t, _, _ in sp.spoken] == ["Where should it go?", "Stopped."]
    (_, a0, a1), (_, b0, b1) = sp.spoken
    assert a1 - a0 >= 0.19 and b1 - b0 >= 0.19 and b0 >= a1  # one at a time
    for t, m in samples:
        if a0 + 0.01 < t < a1 - 0.01 or b0 + 0.01 < t < b1 + 0.29:
            assert m, f"not muted at {t - a0:.2f} s"
        if t > b1 + 0.35:
            assert not m, f"still muted at {t - b1:.2f} s after the last reply"
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == [
        "001-where-should-it-go-.wav",
        "002-stopped-.wav",
    ]


class _Loud:
    """A VAD that hears speech in every frame."""

    def prob(self, frame) -> float:
        return 1.0

    def reset(self) -> None:
        pass


class _Asr:
    def transcribe(self, samples, rate, t0):
        from deixis.types import Word

        return [Word("hello", t0, t0 + 0.1)]


class _Mic:
    def __init__(self, n: int) -> None:
        self.n = n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def blocks(self, timeout: float = 1.0):
        for i in range(self.n):
            time.sleep(0.002)
            yield np.full(512, 0.3, np.float32), i * 0.032
        while True:  # then silence, so an utterance can close
            time.sleep(0.002)
            i += 1
            yield np.zeros(512, np.float32), i * 0.032


def test_listener_hears_nothing_while_the_app_speaks():
    muted = [True]
    lis = Listener(_Asr(), _Loud(), capture=lambda: _Mic(200), muted=lambda: muted[0])
    lis.start()
    try:
        time.sleep(0.3)
        assert lis.events.empty() and lis.segmenter.muted  # the robot's voice: dropped
        muted[0] = False
        time.sleep(0.3)
        assert not lis.segmenter.muted
        kinds = []
        while not lis.events.empty():
            kinds.append(lis.events.get()[0])
        assert "speech_start" in kinds  # the person, once the reply is over
    finally:
        lis.close()


class _FakeSd:
    """Enough of sounddevice to see what rate and shape the speech was played at."""

    class PortAudioError(Exception):
        pass

    def __init__(self, rate=48000, channels=2, mono_fails=False):
        self.info = {"default_samplerate": rate, "max_output_channels": channels}
        self.mono_fails = mono_fails
        self.played = []
        self.default = type("d", (), {"device": [None, 7]})()

    def query_devices(self, dev, kind=None):
        return self.info

    def play(self, x, rate, device=None):
        import numpy as _np

        a = _np.asarray(x)
        if self.mono_fails and a.ndim == 1:
            raise self.PortAudioError("Invalid number of channels")
        self.played.append((a.shape, rate, device))

    def wait(self):
        pass


def _speaker_playing_on(fake, monkeypatch, device=None):
    import sys

    from pointdraw.live.speaker import Speaker

    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    sp = Speaker(lambda t: np.zeros(16000, np.float32), play=True, device=device)
    sp._play(np.zeros(16000, np.float32))
    return sp


def test_the_voice_is_resampled_to_what_the_device_takes(monkeypatch):
    """A WASAPI endpoint refuses any rate but its own: the replies went silent when
    --speaker pointed at the headset (rahul, 7 Oct 2026, "Invalid sample rate")."""
    fake = _FakeSd(rate=48000)
    _speaker_playing_on(fake, monkeypatch, device=16)
    ((shape, rate, device),) = fake.played
    assert rate == 48000 and device == 16
    assert shape[0] == 48000  # one second, resampled from 16 kHz


def test_a_device_that_wants_its_own_rate_only_gets_mono_when_it_can(monkeypatch):
    fake = _FakeSd(rate=16000, channels=2)
    _speaker_playing_on(fake, monkeypatch, device=3)
    ((shape, rate, _),) = fake.played
    assert rate == 16000 and len(shape) == 1  # no needless resample, mono accepted


def test_a_device_that_refuses_mono_is_given_every_channel(monkeypatch):
    fake = _FakeSd(rate=48000, channels=2, mono_fails=True)
    _speaker_playing_on(fake, monkeypatch, device=21)
    ((shape, rate, _),) = fake.played
    assert rate == 48000 and shape == (48000, 2)


class _FlakySd(_FakeSd):
    """Fails the first `fails` streams, as a sleeping Bluetooth link does."""

    def __init__(self, fails, **kw):
        super().__init__(**kw)
        self.fails = fails
        self.attempts = 0

    def play(self, x, rate, device=None):
        self.attempts += 1
        if self.attempts <= self.fails:
            raise self.PortAudioError("Unanticipated host error")
        super().play(x, rate, device)


def test_a_sleeping_bluetooth_link_is_tried_again(monkeypatch):
    """The headset refused the first stream after idling (rahul, 7 Oct 2026). A lost reply
    mid-demo is worse than a stutter."""
    fake = _FlakySd(1, rate=48000)
    _speaker_playing_on(fake, monkeypatch, device=20)
    assert fake.attempts == 2 and len(fake.played) == 1


def test_a_speaker_that_will_not_play_falls_back_to_windows(monkeypatch):
    fake = _FlakySd(4, rate=48000)
    _speaker_playing_on(fake, monkeypatch, device=20)
    assert fake.attempts == 5
    ((_, _, device),) = fake.played
    assert device is None  # Windows' own, so the reply is still heard


def test_without_a_chosen_speaker_a_failure_is_raised(monkeypatch):
    """Nothing to fall back to: the caller logs it, as it always did."""
    import pytest

    fake = _FlakySd(9, rate=48000)
    with pytest.raises(fake.PortAudioError):
        _speaker_playing_on(fake, monkeypatch, device=None)
