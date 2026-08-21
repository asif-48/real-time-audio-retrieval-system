
"""
Closed-set exact-song recognition for a fixed library of songs.

This engine is intentionally optimized for the user's real setup:
phone speaker -> room -> cheap microphone.

It is NOT a humming detector. It assumes the played audio is one of the
indexed reference songs.

Recognition combines:
1. noise-whitened constellation peaks,
2. exact Shazam-style landmark hash voting,
3. tolerant single-peak offset voting.

Unlike the earlier version, it does not reject the top song because of an
arbitrary confidence threshold. In closed-set mode, it always returns the
strongest song when usable audio is present.
"""

from __future__ import annotations

import gzip
import logging
import math
import os
import pickle
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter, median_filter


SAMPLE_RATE = 8000
WINDOW_SIZE = 1024
HOP_SIZE = 256
FRAME_SIZE = WINDOW_SIZE

FREQ_MIN_HZ = 100.0
FREQ_MAX_HZ = 3800.0
FREQ_BANDS_HZ = (100.0, 300.0, 600.0, 1200.0, 2400.0, 3800.0)

TIME_BLOCK_FRAMES = 8
DB_PEAKS_PER_BAND_BLOCK = 1
QUERY_PEAKS_PER_BAND_BLOCK = 2
LOCAL_MAX_FREQ_BINS = 9
LOCAL_MAX_TIME_BINS = 5
SPECTRAL_MEDIAN_BINS = 31
MIN_SNR_DB = 6.0
MIN_LOCAL_CONTRAST_DB = 2.0
FALLBACK_MIN_SNR_DB = 4.0

DB_FANOUT = 10
QUERY_FANOUT = 15
TARGET_TIME_SLICES = 5
MIN_PAIR_DELTA_FRAMES = 2
MAX_PAIR_DELTA_FRAMES = int(round(4.0 * SAMPLE_RATE / HOP_SIZE))

FREQ_QUANT_BINS = 2
TIME_QUANT_FRAMES = 2

PEAK_FREQUENCY_TOLERANCE_BINS = 3
OFFSET_TOLERANCE_FRAMES = 2

RECORD_SECONDS = 20
MIN_QUERY_PEAKS = 80
MIN_QUERY_HASHES = 80
FINGERPRINT_VERSION = "closed_set_exact_song_v1"

DB_FILE = "closed_set_database.pkl.gz"
LOG_FILE = "match_history.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def log_match(song_title: str, confidence: float, method: str) -> None:
    logging.info(
        "MATCH | song=%r | confidence=%.2f%% | method=%s",
        song_title,
        confidence,
        method,
    )


def log_no_match(reason: str) -> None:
    logging.info("NO_MATCH | reason=%s", reason)


class AudioProcessor:
    @staticmethod
    def _to_float_mono(audio_data: np.ndarray) -> np.ndarray:
        x = np.asarray(audio_data)

        if x.ndim > 1:
            x = np.mean(x.astype(np.float32), axis=1)

        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            scale = float(max(abs(info.min), info.max))
            x = x.astype(np.float32) / scale
        else:
            x = x.astype(np.float32, copy=False)

        x = np.nan_to_num(x, copy=False)

        if x.size:
            x = x - float(np.mean(x))

        return x

    @classmethod
    def prepare_audio(cls, audio_data: np.ndarray, sr: int) -> np.ndarray:
        x = cls._to_float_mono(audio_data)

        if x.size < 16:
            return np.zeros(0, dtype=np.float32)

        sr = int(round(float(sr)))
        if sr <= 0:
            raise ValueError("Sample rate must be positive")

        if sr != SAMPLE_RATE:
            common = gcd(sr, SAMPLE_RATE)
            x = signal.resample_poly(
                x,
                SAMPLE_RATE // common,
                sr // common,
            ).astype(np.float32)

        robust_scale = float(np.percentile(np.abs(x), 99.5))
        if robust_scale > 1e-8:
            x = np.clip(x / robust_scale, -3.0, 3.0)

        sos = signal.butter(
            4,
            [FREQ_MIN_HZ, FREQ_MAX_HZ],
            btype="bandpass",
            fs=SAMPLE_RATE,
            output="sos",
        )

        try:
            x = signal.sosfiltfilt(sos, x)
        except ValueError:
            x = signal.sosfilt(sos, x)

        return np.asarray(x, dtype=np.float32)

    @classmethod
    def compute_peaks(
        cls,
        audio_data: np.ndarray,
        sr: int = SAMPLE_RATE,
        *,
        query_mode: bool = False,
    ) -> np.ndarray:
        """
        Extract a controlled-density, noise-whitened constellation map.

        Returns int32 array with shape (N, 2):
            column 0 = STFT time frame
            column 1 = FFT frequency bin
        """
        x = cls.prepare_audio(audio_data, sr)

        if x.size < WINDOW_SIZE:
            return np.empty((0, 2), dtype=np.int32)

        freqs, _, zxx = signal.stft(
            x,
            fs=SAMPLE_RATE,
            window="hann",
            nperseg=WINDOW_SIZE,
            noverlap=WINDOW_SIZE - HOP_SIZE,
            nfft=WINDOW_SIZE,
            boundary=None,
            padded=False,
        )

        magnitude_db = 20.0 * np.log10(
            np.abs(zxx).astype(np.float32) + 1e-10
        )

        frequency_mask = (
            (freqs >= FREQ_MIN_HZ)
            & (freqs <= FREQ_MAX_HZ)
        )

        full_fft_bins = np.flatnonzero(frequency_mask)
        band_frequencies = freqs[frequency_mask]
        magnitude_db = magnitude_db[frequency_mask]

        if magnitude_db.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        # Remove stationary frequency-specific noise.
        noise_floor = np.percentile(
            magnitude_db,
            20.0,
            axis=1,
            keepdims=True,
        )
        snr_map = magnitude_db - noise_floor

        # Remove smooth phone-speaker / cheap-mic frequency coloration.
        spectral_envelope = median_filter(
            magnitude_db,
            size=(SPECTRAL_MEDIAN_BINS, 1),
            mode="nearest",
        )
        local_contrast = magnitude_db - spectral_envelope

        score = snr_map + 0.60 * local_contrast

        local_maximum = maximum_filter(
            score,
            size=(LOCAL_MAX_FREQ_BINS, LOCAL_MAX_TIME_BINS),
            mode="nearest",
        )

        candidate_mask = (
            (score >= local_maximum - 1e-6)
            & (snr_map >= MIN_SNR_DB)
            & (local_contrast >= MIN_LOCAL_CONTRAST_DB)
        )

        peaks_per_zone = (
            QUERY_PEAKS_PER_BAND_BLOCK
            if query_mode
            else DB_PEAKS_PER_BAND_BLOCK
        )

        selected_peaks: List[Tuple[int, int, float]] = []
        time_frame_count = score.shape[1]

        for time_start in range(
            0,
            time_frame_count,
            TIME_BLOCK_FRAMES,
        ):
            time_end = min(
                time_frame_count,
                time_start + TIME_BLOCK_FRAMES,
            )

            for low_hz, high_hz in zip(
                FREQ_BANDS_HZ[:-1],
                FREQ_BANDS_HZ[1:],
            ):
                local_frequency_rows = np.flatnonzero(
                    (band_frequencies >= low_hz)
                    & (band_frequencies < high_hz)
                )

                if local_frequency_rows.size == 0:
                    continue

                coordinates = np.argwhere(
                    candidate_mask[
                        local_frequency_rows,
                        time_start:time_end,
                    ]
                )

                if coordinates.size:
                    values = np.array(
                        [
                            score[
                                local_frequency_rows[row],
                                time_start + column,
                            ]
                            for row, column in coordinates
                        ],
                        dtype=np.float32,
                    )

                    chosen = np.argsort(values)[::-1][
                        :peaks_per_zone
                    ]

                    for chosen_index in chosen:
                        row, column = coordinates[chosen_index]
                        local_row = int(
                            local_frequency_rows[row]
                        )
                        selected_peaks.append(
                            (
                                time_start + int(column),
                                int(full_fft_bins[local_row]),
                                float(values[chosen_index]),
                            )
                        )
                else:
                    # Fallback for heavily damaged regions.
                    zone_scores = score[
                        local_frequency_rows,
                        time_start:time_end,
                    ]

                    chosen = np.argsort(
                        zone_scores.ravel()
                    )[::-1][:peaks_per_zone]

                    for flat_index in chosen:
                        row, column = np.unravel_index(
                            flat_index,
                            zone_scores.shape,
                        )
                        local_row = int(
                            local_frequency_rows[row]
                        )
                        time_column = time_start + int(column)

                        if (
                            snr_map[local_row, time_column]
                            < FALLBACK_MIN_SNR_DB
                        ):
                            continue

                        selected_peaks.append(
                            (
                                time_column,
                                int(full_fft_bins[local_row]),
                                float(zone_scores[row, column]),
                            )
                        )

        strongest: Dict[Tuple[int, int], float] = {}

        for time_frame, frequency_bin, strength in selected_peaks:
            key = (int(time_frame), int(frequency_bin))
            if strength > strongest.get(key, -np.inf):
                strongest[key] = float(strength)

        ordered = sorted(strongest)

        if not ordered:
            return np.empty((0, 2), dtype=np.int32)

        return np.asarray(ordered, dtype=np.int32)

    @staticmethod
    def _pack_hash(
        first_frequency: int,
        second_frequency: int,
        time_delta: int,
    ) -> int:
        return (
            (int(first_frequency) << 19)
            | (int(second_frequency) << 9)
            | int(time_delta)
        )

    @classmethod
    def hashes_from_peaks(
        cls,
        peaks: np.ndarray,
        *,
        query_mode: bool = False,
    ) -> np.ndarray:
        """
        Build landmark pairs.

        Returns int32 array with shape (N, 2):
            column 0 = packed landmark hash
            column 1 = anchor time frame
        """
        peaks = np.asarray(peaks, dtype=np.int32)

        if peaks.ndim != 2 or peaks.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int32)

        fanout = QUERY_FANOUT if query_mode else DB_FANOUT
        result: List[Tuple[int, int]] = []
        seen = set()
        peak_count = len(peaks)

        for anchor_index in range(peak_count):
            anchor_time = int(peaks[anchor_index, 0])
            anchor_frequency = int(peaks[anchor_index, 1])

            candidates: List[Tuple[int, int]] = []
            target_index = anchor_index + 1

            while target_index < peak_count:
                target_time = int(peaks[target_index, 0])
                target_frequency = int(peaks[target_index, 1])
                time_delta = target_time - anchor_time

                if time_delta > MAX_PAIR_DELTA_FRAMES:
                    break

                if time_delta >= MIN_PAIR_DELTA_FRAMES:
                    candidates.append(
                        (target_frequency, time_delta)
                    )

                target_index += 1

            if not candidates:
                continue

            selected: List[Tuple[int, int]] = []
            full_span = (
                MAX_PAIR_DELTA_FRAMES
                - MIN_PAIR_DELTA_FRAMES
                + 1
            )
            per_slice = max(
                1,
                math.ceil(fanout / TARGET_TIME_SLICES),
            )

            for slice_index in range(TARGET_TIME_SLICES):
                low = (
                    MIN_PAIR_DELTA_FRAMES
                    + slice_index
                    * full_span
                    / TARGET_TIME_SLICES
                )
                high = (
                    MIN_PAIR_DELTA_FRAMES
                    + (slice_index + 1)
                    * full_span
                    / TARGET_TIME_SLICES
                )

                bucket = [
                    candidate
                    for candidate in candidates
                    if low <= candidate[1] < high
                ]

                # Deterministic, frequency-diverse selection.
                bucket.sort(
                    key=lambda item: (
                        abs(item[0] - anchor_frequency),
                        item[1],
                    ),
                    reverse=True,
                )
                selected.extend(bucket[:per_slice])

            selected = selected[:fanout]
            quantized_first = int(
                round(anchor_frequency / FREQ_QUANT_BINS)
            )

            for target_frequency, time_delta in selected:
                quantized_second = int(
                    round(target_frequency / FREQ_QUANT_BINS)
                )
                quantized_delta = int(
                    round(time_delta / TIME_QUANT_FRAMES)
                )

                if not (0 <= quantized_delta <= 0x1FF):
                    continue

                packed_hash = cls._pack_hash(
                    quantized_first,
                    quantized_second,
                    quantized_delta,
                )
                pair = (packed_hash, anchor_time)

                if pair in seen:
                    continue

                seen.add(pair)
                result.append(pair)

        if not result:
            return np.empty((0, 2), dtype=np.int32)

        return np.asarray(result, dtype=np.int32)

    @classmethod
    def extract(
        cls,
        audio_data: np.ndarray,
        sr: int,
        *,
        query_mode: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        peaks = cls.compute_peaks(
            audio_data,
            sr,
            query_mode=query_mode,
        )
        hashes = cls.hashes_from_peaks(
            peaks,
            query_mode=query_mode,
        )
        return peaks, hashes

    @staticmethod
    def frequency_bin_to_hz(frequency_bin: int) -> float:
        return (
            float(frequency_bin)
            * SAMPLE_RATE
            / WINDOW_SIZE
        )


def save_database(database: dict, path: str = DB_FILE) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")

    with gzip.open(temporary, "wb", compresslevel=6) as handle:
        pickle.dump(
            database,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(temporary, target)


def load_database(path: str = DB_FILE) -> dict:
    with gzip.open(path, "rb") as handle:
        database = pickle.load(handle)

    if not isinstance(database, dict):
        raise ValueError("Database root is invalid")

    if database.get("version") != FINGERPRINT_VERSION:
        raise ValueError(
            "Database fingerprint version is incompatible. "
            "Run build_database.py once."
        )

    songs = database.get("songs")
    if not isinstance(songs, list) or not songs:
        raise ValueError("Database contains no songs")

    return database


class ClosedSetMatcher:
    """
    Closed-set matcher.

    Since every query is guaranteed to be one of the indexed songs, the system
    ranks every song and returns the strongest one. It does not reject a correct
    winner because a hand-written percentage threshold was not reached.
    """

    def __init__(self, database: dict):
        self.database = database
        self.songs = database["songs"]

        self._peak_index = defaultdict(list)
        self._song_hash_maps: Dict[int, Dict[int, np.ndarray]] = {}

        for song_id, song in enumerate(self.songs):
            peaks = np.asarray(song["peaks"], dtype=np.int32)
            for time_frame, frequency_bin in peaks:
                self._peak_index[int(frequency_bin)].append(
                    (song_id, int(time_frame))
                )

    @staticmethod
    def _smoothed_counter_peak(
        counter: Counter,
        tolerance: int = OFFSET_TOLERANCE_FRAMES,
    ) -> Tuple[int, float]:
        if not counter:
            return 0, 0.0

        best_offset = 0
        best_score = -1.0

        for center in counter:
            score = sum(
                counter.get(center + delta, 0.0)
                for delta in range(
                    -tolerance,
                    tolerance + 1,
                )
            )

            if score > best_score:
                best_score = float(score)
                best_offset = int(center)

        return best_offset, best_score

    def _global_peak_scores(
        self,
        query_peaks: np.ndarray,
    ) -> List[dict]:
        events = defaultdict(list)

        for query_id, (
            query_time,
            query_frequency,
        ) in enumerate(query_peaks):
            query_time = int(query_time)
            query_frequency = int(query_frequency)

            for frequency_delta in range(
                -PEAK_FREQUENCY_TOLERANCE_BINS,
                PEAK_FREQUENCY_TOLERANCE_BINS + 1,
            ):
                weight = 1.0 - 0.15 * abs(frequency_delta)

                if weight <= 0:
                    continue

                for song_id, database_time in self._peak_index.get(
                    query_frequency + frequency_delta,
                    (),
                ):
                    events[song_id].append(
                        (
                            database_time - query_time,
                            query_time,
                            weight,
                            query_id,
                        )
                    )

        rows = []

        for song_id in range(len(self.songs)):
            song_events = events.get(song_id, [])

            if not song_events:
                rows.append(
                    {
                        "song_id": song_id,
                        "peak_votes": 0.0,
                        "peak_anchors": 0,
                        "peak_span_seconds": 0.0,
                    }
                )
                continue

            histogram = Counter()

            for offset, _, weight, _ in song_events:
                histogram[offset] += weight

            best_offset, _ = self._smoothed_counter_peak(
                histogram
            )

            strongest_by_query_peak = {}
            aligned_query_times = set()

            for (
                offset,
                query_time,
                weight,
                query_id,
            ) in song_events:
                if (
                    abs(offset - best_offset)
                    <= OFFSET_TOLERANCE_FRAMES
                ):
                    if (
                        weight
                        > strongest_by_query_peak.get(query_id, 0.0)
                    ):
                        strongest_by_query_peak[query_id] = weight
                    aligned_query_times.add(query_time)

            peak_votes = float(
                sum(strongest_by_query_peak.values())
            )
            peak_anchors = len(strongest_by_query_peak)

            if len(aligned_query_times) >= 2:
                peak_span_seconds = (
                    max(aligned_query_times)
                    - min(aligned_query_times)
                ) * HOP_SIZE / SAMPLE_RATE
            else:
                peak_span_seconds = 0.0

            rows.append(
                {
                    "song_id": song_id,
                    "peak_votes": peak_votes,
                    "peak_anchors": peak_anchors,
                    "peak_span_seconds": float(
                        peak_span_seconds
                    ),
                }
            )

        return rows

    def _song_hash_map(
        self,
        song_id: int,
    ) -> Dict[int, np.ndarray]:
        cached = self._song_hash_maps.get(song_id)

        if cached is not None:
            return cached

        hashes = np.asarray(
            self.songs[song_id]["hashes"],
            dtype=np.int32,
        )

        temporary = defaultdict(list)

        for packed_hash, anchor_time in hashes:
            temporary[int(packed_hash)].append(
                int(anchor_time)
            )

        mapping = {
            packed_hash: np.asarray(
                times,
                dtype=np.int32,
            )
            for packed_hash, times in temporary.items()
        }

        # Keep only a small LRU-like cache to control memory for 52 songs.
        if len(self._song_hash_maps) >= 12:
            oldest_key = next(iter(self._song_hash_maps))
            self._song_hash_maps.pop(oldest_key, None)

        self._song_hash_maps[song_id] = mapping
        return mapping

    def _hash_score_for_song(
        self,
        query_hashes: np.ndarray,
        song_id: int,
    ) -> dict:
        song_hash_map = self._song_hash_map(song_id)
        histogram = Counter()

        query_matches = []

        for query_id, (
            packed_hash,
            query_time,
        ) in enumerate(query_hashes):
            database_times = song_hash_map.get(
                int(packed_hash)
            )

            if database_times is None:
                continue

            query_time = int(query_time)

            for database_time in database_times:
                offset = int(database_time) - query_time
                histogram[offset] += 1
                query_matches.append(
                    (
                        offset,
                        query_time,
                        query_id,
                    )
                )

        best_offset, hash_votes = self._smoothed_counter_peak(
            histogram
        )

        aligned_query_ids = set()
        aligned_query_times = set()

        for offset, query_time, query_id in query_matches:
            if (
                abs(offset - best_offset)
                <= OFFSET_TOLERANCE_FRAMES
            ):
                aligned_query_ids.add(query_id)
                aligned_query_times.add(query_time)

        if len(aligned_query_times) >= 2:
            span_seconds = (
                max(aligned_query_times)
                - min(aligned_query_times)
            ) * HOP_SIZE / SAMPLE_RATE
        else:
            span_seconds = 0.0

        return {
            "hash_votes": float(hash_votes),
            "hash_anchors": len(aligned_query_ids),
            "hash_span_seconds": float(span_seconds),
        }

    def match(
        self,
        query_peaks: np.ndarray,
        query_hashes: np.ndarray,
    ) -> Tuple[dict | None, List[dict]]:
        query_peaks = np.asarray(
            query_peaks,
            dtype=np.int32,
        )
        query_hashes = np.asarray(
            query_hashes,
            dtype=np.int32,
        )

        if len(query_peaks) < MIN_QUERY_PEAKS:
            log_no_match(
                f"too few query peaks: {len(query_peaks)}"
            )
            return None, []

        peak_rows = self._global_peak_scores(query_peaks)

        # Compute exact landmark evidence for every song. For only 52 songs,
        # this is still practical and avoids missing the correct song because
        # of a preliminary shortlist.
        rows = []

        for peak_row in peak_rows:
            song_id = int(peak_row["song_id"])
            hash_row = self._hash_score_for_song(
                query_hashes,
                song_id,
            )

            # Pair hashes are much more selective than individual peaks.
            ranking_score = (
                2.5 * hash_row["hash_votes"]
                + peak_row["peak_votes"]
                + 0.20 * peak_row["peak_anchors"]
                + 0.10 * hash_row["hash_anchors"]
            )

            song = self.songs[song_id]

            rows.append(
                {
                    "song_id": song_id,
                    "title": song["title"],
                    "artist": song.get(
                        "artist",
                        "Unknown",
                    ),
                    "url": song.get("url", ""),
                    "ranking_score": float(ranking_score),
                    **peak_row,
                    **hash_row,
                }
            )

        rows.sort(
            key=lambda row: row["ranking_score"],
            reverse=True,
        )

        if not rows:
            log_no_match("no candidates")
            return None, []

        best = rows[0]
        second = (
            rows[1]
            if len(rows) > 1
            else {
                "ranking_score": 0.0,
                "hash_votes": 0.0,
                "peak_votes": 0.0,
            }
        )

        ratio = (
            best["ranking_score"]
            / max(second["ranking_score"], 1e-9)
        )

        # Closed-set confidence: relative separation plus absolute evidence.
        # This is a display score, not a statistical probability.
        ratio_component = min(
            1.0,
            max(0.0, (ratio - 1.0) / 3.0),
        )
        hash_component = min(
            1.0,
            best["hash_votes"] / 80.0,
        )
        peak_component = min(
            1.0,
            best["peak_votes"] / 160.0,
        )
        span_component = min(
            1.0,
            max(
                best["hash_span_seconds"],
                best["peak_span_seconds"],
            )
            / 12.0,
        )

        confidence = 100.0 * (
            0.35 * ratio_component
            + 0.35 * hash_component
            + 0.20 * peak_component
            + 0.10 * span_component
        )

        # The correct closed-set winner must still be displayed even if a
        # difficult recording receives a modest display confidence.
        best["confidence"] = float(
            max(1.0, min(100.0, confidence))
        )
        best["winner_ratio"] = float(ratio)
        best["second_title"] = second.get("title", "")
        best["second_score"] = float(
            second.get("ranking_score", 0.0)
        )

        for row in rows[1:]:
            row["confidence"] = float(
                100.0
                * row["ranking_score"]
                / max(best["ranking_score"], 1e-9)
            )

        logging.info(
            "CLOSED_SET_MATCH | peaks=%d hashes=%d | "
            "winner=%r score=%.1f ratio=%.2f "
            "hash_votes=%.1f peak_votes=%.1f "
            "hash_span=%.2fs peak_span=%.2fs | "
            "second=%r score=%.1f",
            len(query_peaks),
            len(query_hashes),
            best["title"],
            best["ranking_score"],
            best["winner_ratio"],
            best["hash_votes"],
            best["peak_votes"],
            best["hash_span_seconds"],
            best["peak_span_seconds"],
            best["second_title"],
            best["second_score"],
        )

        return best, rows[1:5]
