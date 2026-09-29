"""Mapping PortAudio block times onto the shared clock."""

import pytest

from pointdraw.audio.capture import BlockClock

SR = 16000
N = 512  # samples per block
BLOCK_S = N / SR


def test_raw_estimate_uses_adc_time_when_valid():
    # PortAudio says the first sample hit the ADC 40 ms before "current time".
    raw = BlockClock.raw_estimate(N, SR, now=10.0, adc_time=5.00, current_time=5.04)
    assert raw == pytest.approx(9.96)


@pytest.mark.parametrize("adc,cur", [(0.0, 5.0), (None, None), (6.0, 5.0), (1.0, 9.0)])
def test_raw_estimate_falls_back_without_a_usable_adc_time(adc, cur):
    raw = BlockClock.raw_estimate(
        N, SR, now=10.0, adc_time=adc, current_time=cur, input_latency=0.01
    )
    assert raw == pytest.approx(10.0 - BLOCK_S - 0.01)


def test_stamps_are_continuous_despite_callback_jitter():
    c = BlockClock(sample_rate=SR)
    jitter = [0.0, 0.004, -0.003, 0.008, -0.006, 0.002, 0.0, 0.005]
    stamps = [c.stamp(N, 50.0 + k * BLOCK_S + j) for k, j in enumerate(jitter)]
    steps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    for s in steps:
        assert s == pytest.approx(BLOCK_S, abs=0.0005)
    assert stamps[0] == pytest.approx(50.0)
    assert c.resyncs == 0


def test_resyncs_after_a_gap():
    c = BlockClock(sample_rate=SR)
    for k in range(5):
        c.stamp(N, 1.0 + k * BLOCK_S)
    # The stream stalled for 0.5 s (e.g. device glitch): the next block really is later.
    t = c.stamp(N, 1.0 + 5 * BLOCK_S + 0.5)
    assert t == pytest.approx(1.0 + 5 * BLOCK_S + 0.5, abs=1e-6)
    assert c.resyncs == 1


def test_tracks_slow_drift():
    c = BlockClock(sample_rate=SR)
    drift = 0.0001  # the sound card clock runs slightly slow: 0.1 ms extra per block
    for k in range(400):
        t = c.stamp(N, 2.0 + k * (BLOCK_S + drift))
    assert t == pytest.approx(2.0 + 399 * (BLOCK_S + drift), abs=0.01)
