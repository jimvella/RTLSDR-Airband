"""
Output validator for RTLSDR-Airband system tests.

Level 2 validation: verifies MP3 duration and audio properties against
expected values.

MP3 format: LAME-encoded, mono, 8000 Hz sample rate, VBR with LAME tag.
rtl_airband appends a date+hour timestamp to the filename_template, so MP3
files are found via glob. Each test writes to its own test_output/<name>/
subdirectory, so filenames are unique within a run.
"""

import re
import wave
from pathlib import Path

import numpy as np
from mutagen.mp3 import MP3

# rtl_airband hardcodes MP3_RATE = 8000 Hz for all MP3 output
_MP3_SAMPLE_RATE = 8000


def _template_mp3s(mp3_dir: Path, filename_template: str) -> list[Path]:
    """The MP3 files written for this template, in name order."""
    return sorted(mp3_dir.glob(f"{filename_template}_[0-9]*.mp3"))


def validate_mp3(
    mp3_dir: Path,
    filename_template: str,
    expected_duration_s: float,
    tolerance: float,
) -> Path:
    """
    Assert that an MP3 output exists with expected duration and valid audio properties.

    rtl_airband appends a date+hour timestamp to the filename_template, so this
    function uses a glob to locate the file. Returns the path to the validated file.

    Expected audio properties (from rtl_airband's hardcoded LAME settings):
      - Sample rate: 8000 Hz (MP3_RATE)
      - Mode: mono
      - Encoding: VBR with LAME tag (accurate duration in metadata)

    Args:
        mp3_dir: Directory containing the MP3 output files.
        filename_template: Filename template used in the config (without extension).
        expected_duration_s: Expected audio duration in seconds.
        tolerance: Allowed fractional deviation from expected duration (default ±20%).

    Returns:
        Path to the validated MP3 file.

    Raises:
        AssertionError: If no file is found, duration is out of range, or audio
                        properties (sample rate, bitrate) are invalid.
    """
    matches = _template_mp3s(mp3_dir, filename_template)
    assert (
        matches
    ), f"No .mp3 output file found matching '{filename_template}_[0-9]*.mp3' in {mp3_dir}"

    # Validate each file and sum durations (multiple files arise from hour-boundary splits)
    actual_s = 0.0
    for mp3_file in matches:
        assert mp3_file.stat().st_size > 0, f"MP3 file is empty: {mp3_file.name}"
        audio = MP3(mp3_file)
        actual_s += audio.info.length

        # Sample rate — hardcoded to 8000 Hz in rtl_airband (MP3_RATE)
        assert audio.info.sample_rate == _MP3_SAMPLE_RATE, (
            f"Expected MP3 sample rate {_MP3_SAMPLE_RATE} Hz, "
            f"got {audio.info.sample_rate} Hz: {mp3_file.name}"
        )

        # Bitrate — VBR so this is the average; must be positive (file has audio content)
        assert (
            audio.info.bitrate > 0
        ), f"MP3 average bitrate is 0 kbps (empty or corrupt file): {mp3_file.name}"

    label = (
        matches[0].name
        if len(matches) == 1
        else f"{len(matches)} files (hour boundary)"
    )
    deviation = abs(actual_s - expected_duration_s) / expected_duration_s
    print(
        f"mp3 {label}: "
        f"actual={actual_s:.2f}s expected={expected_duration_s:.2f}s "
        f"deviation={deviation*100:.1f}% (limit ±{tolerance*100:.0f}%)"
    )
    assert deviation <= tolerance, (
        f"MP3 duration {actual_s:.2f}s deviates {deviation:.1%} from "
        f"expected {expected_duration_s:.2f}s (±{tolerance:.0%}): {label}"
    )

    return matches[0]


def validate_mp3_range(
    mp3_dir: Path,
    filename_template: str,
    min_duration_s: float,
    max_duration_s: float,
) -> Path:
    """Assert an MP3 output exists with duration in [min_duration_s, max_duration_s].

    Use this when the expected duration isn't a single value but a band — e.g.
    a squelch-disabled run that always passes the signal but may also pass the
    surrounding noise pad depending on tracker convergence.

    Sample-rate and bitrate checks match validate_mp3(); only the duration
    assertion differs.
    """
    matches = _template_mp3s(mp3_dir, filename_template)
    assert (
        matches
    ), f"No .mp3 output file found matching '{filename_template}_[0-9]*.mp3' in {mp3_dir}"

    actual_s = 0.0
    for mp3_file in matches:
        assert mp3_file.stat().st_size > 0, f"MP3 file is empty: {mp3_file.name}"
        audio = MP3(mp3_file)
        actual_s += audio.info.length
        assert audio.info.sample_rate == _MP3_SAMPLE_RATE, (
            f"Expected MP3 sample rate {_MP3_SAMPLE_RATE} Hz, "
            f"got {audio.info.sample_rate} Hz: {mp3_file.name}"
        )
        assert (
            audio.info.bitrate > 0
        ), f"MP3 average bitrate is 0 kbps (empty or corrupt file): {mp3_file.name}"

    label = (
        matches[0].name
        if len(matches) == 1
        else f"{len(matches)} files (hour boundary)"
    )
    print(
        f"mp3 {label}: actual={actual_s:.2f}s "
        f"expected range=[{min_duration_s:.2f}, {max_duration_s:.2f}]s"
    )
    assert min_duration_s <= actual_s <= max_duration_s, (
        f"MP3 duration {actual_s:.2f}s is outside expected range "
        f"[{min_duration_s:.2f}, {max_duration_s:.2f}]s: {label}"
    )

    return matches[0]


def count_mp3_files(mp3_dir: Path, filename_template: str) -> int:
    """
    Count the non-empty MP3 files written for this template.

    split_on_transmission outputs produce one file per transmission, so the count
    is what distinguishes a file that stayed open across a gap from one that split.
    """
    matches = _template_mp3s(mp3_dir, filename_template)
    for mp3_file in matches:
        assert mp3_file.stat().st_size > 0, f"MP3 file is empty: {mp3_file.name}"
    return len(matches)


def assert_mp3_present(mp3_dir: Path, filename_template: str) -> None:
    """
    Assert that at least one non-empty MP3 file exists for this template.

    Used when duration validation is skipped (e.g. mixer output at high speedup).

    Raises:
        AssertionError: If no matching file is found or all matching files are empty.
    """
    assert count_mp3_files(
        mp3_dir, filename_template
    ), f"No .mp3 output file found matching '{filename_template}_[0-9]*.mp3' in {mp3_dir}"


def assert_mp3_silent(mp3_dir: Path, filename_template: str) -> None:
    """
    Assert that no non-empty MP3 file was created for this template.

    With non-continuous MP3 output (rtl_airband default), no file is created when
    the squelch stays closed for the entire run. Used for squelch-closed and
    wrong-CTCSS tests.

    Args:
        mp3_dir: Directory that would contain MP3 output files.
        filename_template: Filename template used in the config (without extension).

    Raises:
        AssertionError: If a non-empty MP3 file is found.
    """
    matches = _template_mp3s(mp3_dir, filename_template)
    if not matches:
        return  # No file created — correct for squelch-closed

    for mp3_file in matches:
        assert mp3_file.stat().st_size == 0, (
            f"Expected no MP3 output (squelch/CTCSS gate closed), but "
            f"{mp3_file.name} contains {mp3_file.stat().st_size} bytes"
        )


def split_file_names(
    mp3_dir: Path,
    filename_template: str,
    freq_hz: int | None = None,
    milliseconds: bool = False,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Sort a template's MP3 files by split_include_transmission_start naming.

    Returns (standalone stamps, {transmission start stamp: part start stamps in
    file-name order}). Any other name for the template fails the assertion. Pass
    freq_hz for an include_freq output; every name must then end with it. Pass
    milliseconds for a split_timestamp_milliseconds output; every stamp must then
    end with ".mmm".
    """
    stamp = r"(\d{8}_\d{6}\.\d{3})" if milliseconds else r"(\d{8}_\d{6})"
    freq = f"_{freq_hz}" if freq_hz is not None else ""
    pattern = re.compile(
        rf"{re.escape(filename_template)}_{stamp}(?:_{stamp})?{freq}\.mp3"
    )
    standalone: list[str] = []
    parts: dict[str, list[str]] = {}
    for mp3_file in _template_mp3s(mp3_dir, filename_template):
        match = pattern.fullmatch(mp3_file.name)
        assert match, f"Unexpected file name for '{filename_template}': {mp3_file.name}"
        assert mp3_file.stat().st_size > 0, f"MP3 file is empty: {mp3_file.name}"
        transmission_stamp, part_stamp = match.groups()
        if part_stamp is None:
            standalone.append(transmission_stamp)
        else:
            parts.setdefault(transmission_stamp, []).append(part_stamp)
    return standalone, parts


def total_mp3_duration_s(mp3_dir: Path, filename_template: str) -> float:
    """Sum the audio duration of every MP3 file written for this template."""
    return sum(MP3(f).info.length for f in _template_mp3s(mp3_dir, filename_template))


def template_wavs(wav_dir: Path, filename_template: str) -> list[Path]:
    """The WAV files written for this template, in name order."""
    return sorted(wav_dir.glob(f"{filename_template}_[0-9]*.wav"))


def read_wav(path: Path) -> tuple[int, int, np.ndarray]:
    """
    Read a WAV written by a file output and check its header.

    Asserts 16-bit PCM and that the header's sizes match the file, which shows the
    file was finalized on close. Returns (sample rate, channels, samples as float
    in [-1, 1], shape (frames, channels)).
    """
    with wave.open(str(path), "rb") as wav:
        assert wav.getsampwidth() == 2, f"Expected 16-bit samples: {path.name}"
        rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    assert path.stat().st_size == 44 + frames * 2 * channels, (
        f"WAV header sizes ({frames} frames) do not match the file size "
        f"{path.stat().st_size}: {path.name}"
    )
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    return rate, channels, samples.reshape(-1, channels)


def wav_duration_s(path: Path) -> float:
    rate, _, samples = read_wav(path)
    return len(samples) / rate
