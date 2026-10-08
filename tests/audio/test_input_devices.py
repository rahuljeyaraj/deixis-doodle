"""Choosing the microphone (rahul, 7 Oct 2026).

On the test machine Windows' default input is the webcam's microphone and the USB one on
the desk is a separate entry, so the app listened through the wrong one and said nothing
about it. `--mic` takes an index or any part of a name; a name that matches more than one
host API takes the WASAPI entry, which reports ADC timestamps.
"""

import pytest

from pointdraw.audio import capture

DEVICES = [
    {"index": 1, "name": "Microphone (2- Lenovo FHD Webca", "hostapi": "MME", "channels": 1},
    {
        "index": 7,
        "name": "Microphone (2- Lenovo FHD Webcam Audio)",
        "hostapi": "Windows DirectSound",
        "channels": 2,
    },
    {
        "index": 15,
        "name": "Microphone (2- Lenovo FHD Webcam Audio)",
        "hostapi": "Windows WASAPI",
        "channels": 2,
    },
    {
        "index": 22,
        "name": "Microphone (USBAudio1.0)",
        "hostapi": "Windows WDM-KS",
        "channels": 1,
    },
]


@pytest.fixture
def devices(monkeypatch):
    monkeypatch.setattr(capture, "list_devices", lambda kind="input": DEVICES)
    monkeypatch.setattr(capture, "default_input_device", lambda: 1)
    return DEVICES


@pytest.mark.parametrize(("spec", "want"), [(22, 22), ("22", 22), (" 15 ", 15)])
def test_an_index_is_taken_as_it_is(devices, spec, want):
    assert capture.find_input_device(spec)["index"] == want


@pytest.mark.parametrize(("spec", "want"), [("usb", 22), ("USBAudio", 22), ("usbaudio1.0", 22)])
def test_a_name_matches_whatever_the_case(devices, spec, want):
    assert capture.find_input_device(spec)["index"] == want


def test_a_name_on_several_host_apis_takes_wasapi(devices):
    # ADC timestamps: MME has none, so the words would be stamped by callback time.
    assert capture.find_input_device("lenovo")["index"] == 15


@pytest.mark.parametrize("spec", [99, "ipad", ""])
def test_no_match_names_every_microphone(devices, spec):
    with pytest.raises(ValueError) as e:
        capture.find_input_device(spec)
    assert "USBAudio1.0" in str(e.value) and "Lenovo" in str(e.value)


def test_the_table_marks_windows_default(devices):
    rows = capture.input_device_table().splitlines()
    assert len(rows) == 4
    assert "default" in rows[0] and "Lenovo" in rows[0]
    assert "default" not in rows[-1]


def test_no_microphone_at_all_says_so(monkeypatch):
    monkeypatch.setattr(capture, "list_devices", lambda kind="input": [])
    with pytest.raises(ValueError, match="no microphone found"):
        capture.find_input_device("usb")


OUTPUTS = [
    {"index": 3, "name": "Headphones (OnePlus Bullets Wir", "hostapi": "MME", "channels": 2},
    {
        "index": 9,
        "name": "Headphones (OnePlus Bullets Wireless Z3)",
        "hostapi": "Windows DirectSound",
        "channels": 2,
    },
    {
        "index": 12,
        "name": "Headphones (OnePlus Bullets Wireless Z3)",
        "hostapi": "Windows WASAPI",
        "channels": 2,
    },
    {"index": 13, "name": "BenQ GW2790Q", "hostapi": "Windows WASAPI", "channels": 2},
]


def test_a_speaker_is_found_by_name(monkeypatch):
    """`--speaker oneplus` for the headset: the replies go to the ear, not to the room."""
    monkeypatch.setattr(capture, "list_devices", lambda kind="input": OUTPUTS)
    monkeypatch.setattr(capture, "default_output_device", lambda: 3)
    assert capture.find_output_device("oneplus")["index"] == 12  # WASAPI first
    assert capture.find_output_device("benq")["index"] == 13
    with pytest.raises(ValueError, match="no speaker whose name"):
        capture.find_output_device("sony")
    assert "default" in capture.device_table("output").splitlines()[0]
