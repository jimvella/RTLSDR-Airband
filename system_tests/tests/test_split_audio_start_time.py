"""
test_split_audio_start_time.py — split files are named after when their audio was received.

The fixture holds two transmissions that start exactly 8.000 s apart (DURATION_A_S +
GAP_S). The file input stamps every chunk with its position in the recording, so the
two files' names must be 8.000 s apart whatever the replay speed. The replay runs at
SPEEDUP_FACTOR, where names taken from the output thread's clock would be about
8.000 / SPEEDUP_FACTOR apart instead.

A rawfile output on the same channel is named after iq_out[0], which comes from
AGC_EXTRA (100) audio samples after waveout[0]'s source: 12.5 ms later.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

from conftest import BinaryUnderTest, run_rtl_airband
from helpers import config_writer, iq_generator

SAMPLE_RATE = 2_048_000
CENTERFREQ_HZ = 120_000_000
# Same fixture parameters as test_scan so the cached IQ file is shared.
DURATION_A_S = 5.0
GAP_S = 3.0
DURATION_B_S = 5.0
TRANSMISSION_SPACING_S = DURATION_A_S + GAP_S
TOTAL_IQ_DURATION_S = DURATION_A_S + GAP_S + DURATION_B_S + 2 * iq_generator.NOISE_PAD_S

# Fast enough that clock-based names would be far off, slow enough that each
# transmission (2.5 s) and gap (1.5 s) still clear split_min_file_time (1.0 s)
# and split_max_idle_time (0.5 s), which run on the wall clock.
SPEEDUP_FACTOR = 2.0
TIMEOUT_S = TOTAL_IQ_DURATION_S / SPEEDUP_FACTOR * 3 + 30

# 1 ms from truncating each name to milliseconds, 1 ms for rounding to microseconds
TOLERANCE_S = 0.002
# AGC_EXTRA audio samples at 8 kHz
IQ_AFTER_AUDIO_S = 100 / 8000


def pytest_generate_tests(metafunc):
    """Naming is modulation-independent; run against one binary only."""
    if "binary_under_test" in metafunc.fixturenames:
        am_bins: list[BinaryUnderTest] = metafunc.config._rtlsdr_am_binaries
        metafunc.parametrize(
            "binary_under_test",
            am_bins[:1],
            ids=[b.label for b in am_bins[:1]],
        )


def _stamps(directory: Path, template: str, suffix: str) -> list[float]:
    """Epoch seconds of each file's name, sorted (names are UTC)."""
    stamps = []
    for path in sorted(directory.glob(f"{template}_[0-9]*{suffix}")):
        stamp = path.name[len(template) + 1 : -len(suffix)]
        parsed = datetime.strptime(stamp, "%Y%m%d_%H%M%S.%f")
        stamps.append(parsed.replace(tzinfo=timezone.utc).timestamp())
    return stamps


def test_split_audio_start_time(
    binary_under_test: BinaryUnderTest,
    test_output_dir: Path,
    cache_dir: Path,
) -> None:
    iq_file = iq_generator.get_or_generate_scan(
        duration_a_s=DURATION_A_S,
        gap_s=GAP_S,
        duration_b_s=DURATION_B_S,
        cache_dir=cache_dir,
    )
    config_path = test_output_dir / "rtl_airband.conf"
    config_writer.write_config(
        config_path=config_path,
        iq_filepath=iq_file,
        sample_rate=SAMPLE_RATE,
        centerfreq_hz=CENTERFREQ_HZ,
        channels=[
            {
                "freq_hz": CENTERFREQ_HZ + iq_generator.SCAN_DEMOD_OFFSET_HZ,
                "split_outputs": [
                    {"directory": str(test_output_dir), "template": "audio"},
                    {
                        "directory": str(test_output_dir),
                        "template": "iq",
                        "type": "rawfile",
                    },
                ],
            }
        ],
        output_dir=test_output_dir,
        speedup_factor=SPEEDUP_FACTOR,
        mode="multichannel",
        split_times={"split_timestamp_milliseconds": True},
    )

    launched = time.time()
    run_rtl_airband(binary_under_test.path, config_path, timeout_s=TIMEOUT_S)

    audio = _stamps(test_output_dir, "audio", ".mp3")
    iq = _stamps(test_output_dir, "iq", ".cf32")
    assert len(audio) == 2, f"Expected 2 audio files, got {audio}"
    assert len(iq) == 2, f"Expected 2 rawfile files, got {iq}"

    spacing = audio[1] - audio[0]
    assert abs(spacing - TRANSMISSION_SPACING_S) <= TOLERANCE_S, (
        f"Audio file names are {spacing:.3f}s apart; the transmissions start "
        f"{TRANSMISSION_SPACING_S:.3f}s apart in the recording"
    )

    for audio_stamp, iq_stamp in zip(audio, iq):
        offset = iq_stamp - audio_stamp
        assert abs(offset - IQ_AFTER_AUDIO_S) <= TOLERANCE_S, (
            f"rawfile name is {offset * 1000:.1f} ms after the audio file's, "
            f"expected {IQ_AFTER_AUDIO_S * 1000:.1f} ms"
        )

    # The first transmission starts NOISE_PAD_S into the recording; its file starts at
    # the beginning of the 125 ms audio batch in which the squelch opened. Reading
    # starts shortly after launch.
    first = audio[0] - launched
    assert iq_generator.NOISE_PAD_S - 0.15 <= first <= iq_generator.NOISE_PAD_S + 1.5, (
        f"First file is named {first:.3f}s after launch; the transmission is "
        f"{iq_generator.NOISE_PAD_S}s into the recording"
    )
