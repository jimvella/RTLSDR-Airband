"""
test_wav_output.py — file outputs with format = "wav".

  split:  an mp3 and a wav split output on one channel see the same two
          transmissions 8.000 s apart. The wav files get the same names, valid
          headers and about the same audio. Because WAV has no encoder delay,
          the tone's onset in each file gives when it was received
          (name + onset / rate); the two onsets must be 8.000 s apart.
  append: an hourly wav output written by two runs holds both runs' audio and
          the discontinuity tones between them, with the header updated.
  config: a format other than "mp3" or "wav" is rejected.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from conftest import BinaryUnderTest, run_rtl_airband
from helpers import config_writer, iq_generator, output_validator

SAMPLE_RATE = 2_048_000
CENTERFREQ_HZ = 120_000_000
WAVE_RATE = 8000  # non-NFM build

# Same fixture parameters as test_scan so the cached IQ file is shared.
DURATION_A_S = 5.0
GAP_S = 3.0
DURATION_B_S = 5.0
TRANSMISSION_SPACING_S = DURATION_A_S + GAP_S
SPLIT_SPEEDUP = 2.0  # see test_split_audio_start_time.py
SPLIT_TIMEOUT_S = 15.0 / SPLIT_SPEEDUP * 3 + 30

# Same fixture as test_am_squelch_open.
AM_OFFSET_HZ = 25_000
AM_TONE_HZ = 1_000
AM_DURATION_S = 10.0
APPEND_SPEEDUP = 4.0
APPEND_TIMEOUT_S = 12.0 / APPEND_SPEEDUP * 3 + 30
DISCONTINUITY_TONES_S = 6 * 0.120

ONSET_LEVEL = 0.1
TOLERANCE_S = 0.002


def pytest_generate_tests(metafunc):
    """The WAV writer is modulation-independent; run against one binary only."""
    if "binary_under_test" in metafunc.fixturenames:
        am_bins: list[BinaryUnderTest] = metafunc.config._rtlsdr_am_binaries
        metafunc.parametrize(
            "binary_under_test",
            am_bins[:1],
            ids=[b.label for b in am_bins[:1]],
        )


def _stamp_s(path: Path, template: str) -> float:
    stamp = path.name[len(template) + 1 : -len(path.suffix)]
    parsed = datetime.strptime(stamp, "%Y%m%d_%H%M%S.%f")
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def test_wav_split(
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
                    {"directory": str(test_output_dir), "template": "mp3"},
                    {
                        "directory": str(test_output_dir),
                        "template": "wav",
                        "overrides": {"format": "wav"},
                    },
                ],
            }
        ],
        output_dir=test_output_dir,
        speedup_factor=SPLIT_SPEEDUP,
        mode="multichannel",
        split_times={"split_timestamp_milliseconds": True},
    )
    run_rtl_airband(binary_under_test.path, config_path, timeout_s=SPLIT_TIMEOUT_S)

    mp3s = sorted(test_output_dir.glob("mp3_[0-9]*.mp3"))
    wavs = output_validator.template_wavs(test_output_dir, "wav")
    assert len(wavs) == 2, f"Expected 2 wav files, got {[w.name for w in wavs]}"
    assert [_stamp_s(w, "wav") for w in wavs] == [
        _stamp_s(m, "mp3") for m in mp3s
    ], f"wav and mp3 names differ: {wavs} vs {mp3s}"

    onsets = []
    for wav in wavs:
        rate, channels, samples = output_validator.read_wav(wav)
        assert (rate, channels) == (
            WAVE_RATE,
            1,
        ), f"{wav.name}: {rate} Hz, {channels} ch"
        loud = np.flatnonzero(np.abs(samples[:, 0]) > ONSET_LEVEL)
        assert loud.size, f"{wav.name} has no audio above {ONSET_LEVEL}"
        onsets.append(_stamp_s(wav, "wav") + loud[0] / rate)

    wav_s = sum(output_validator.wav_duration_s(w) for w in wavs)
    mp3_s = output_validator.total_mp3_duration_s(test_output_dir, "mp3")
    assert (
        abs(wav_s - mp3_s) <= 0.1 * mp3_s
    ), f"wav files hold {wav_s:.2f}s, mp3 files {mp3_s:.2f}s"

    spacing = onsets[1] - onsets[0]
    assert abs(spacing - TRANSMISSION_SPACING_S) <= TOLERANCE_S, (
        f"Tone onsets are {spacing:.4f}s apart by name + offset; the transmissions "
        f"start {TRANSMISSION_SPACING_S:.3f}s apart in the recording"
    )


def _append_config(iq_file: Path, directory: Path) -> str:
    return f"""devices: ({{
  type = "file"; filepath = "{iq_file}"; sample_rate = {SAMPLE_RATE};
  centerfreq = {CENTERFREQ_HZ}; speedup_factor = {APPEND_SPEEDUP};
  channels: ({{
    freq = {(CENTERFREQ_HZ + AM_OFFSET_HZ) / 1e6:.6f};
    outputs: ({{ type = "file"; directory = "{directory}"; filename_template = "hourly";
      format = "wav"; append = true; }});
  }});
}});
"""


def test_wav_append(
    binary_under_test: BinaryUnderTest,
    test_output_dir: Path,
    cache_dir: Path,
) -> None:
    iq_file = iq_generator.get_or_generate_am(
        offset_hz=AM_OFFSET_HZ,
        audio_hz=AM_TONE_HZ,
        duration_s=AM_DURATION_S,
        cache_dir=cache_dir,
    )
    config_path = test_output_dir / "rtl_airband.conf"
    config_path.write_text(_append_config(iq_file, test_output_dir))
    for _ in range(2):
        run_rtl_airband(binary_under_test.path, config_path, timeout_s=APPEND_TIMEOUT_S)

    wavs = output_validator.template_wavs(test_output_dir, "hourly")
    total_s = sum(output_validator.wav_duration_s(w) for w in wavs)
    if len(wavs) == 1:
        expected_s = 2 * AM_DURATION_S + DISCONTINUITY_TONES_S
    else:
        # the runs fell either side of an hour: two files, no appending
        assert len(wavs) == 2, f"Unexpected files: {[w.name for w in wavs]}"
        expected_s = 2 * AM_DURATION_S
    assert (
        abs(total_s - expected_s) <= 0.1 * expected_s
    ), f"Expected about {expected_s:.2f}s of audio, got {total_s:.2f}s in {wavs}"


def test_wav_format_config_validation(
    binary_under_test: BinaryUnderTest,
    test_output_dir: Path,
    cache_dir: Path,
) -> None:
    iq_file = iq_generator.get_or_generate_am(
        offset_hz=AM_OFFSET_HZ,
        audio_hz=AM_TONE_HZ,
        duration_s=AM_DURATION_S,
        cache_dir=cache_dir,
    )
    good = _append_config(iq_file, test_output_dir)
    for name, bad in {
        "unknown": good.replace('format = "wav"', 'format = "flac"'),
        "not_string": good.replace('format = "wav"', "format = 1"),
    }.items():
        config_path = test_output_dir / f"{name}.conf"
        config_path.write_text(bad)
        result = run_rtl_airband(
            binary_under_test.path, config_path, timeout_s=APPEND_TIMEOUT_S, check=False
        )
        assert result.returncode != 0, f"{name}: config should be rejected"
        assert (
            "Configuration error" in result.stderr and "format" in result.stderr
        ), f"{name}: unexpected stderr:\n{result.stderr}"
