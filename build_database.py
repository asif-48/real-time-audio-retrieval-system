
"""
Build the closed-set song database from local WAV files.

Normal use:
    python build_database.py --folder downloaded_tracks

For the uploaded 5-song validation set:
    python build_database.py --folder Songs --original-suffix _original

The script:
- never downloads anything,
- skips microphone test recordings,
- keeps completed tracks cached,
- saves after every newly indexed song,
- can reuse titles from an older song_fingerprints JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from dsp_core import (
    AudioProcessor,
    DB_FILE,
    FINGERPRINT_VERSION,
    save_database,
)


EXCLUDED_SUFFIXES = (
    "_good",
    "_noisy",
    "_mic",
    "_test",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        default="downloaded_tracks",
        help="Folder containing reference WAV files",
    )
    parser.add_argument(
        "--database",
        default=DB_FILE,
        help="Output database path",
    )
    parser.add_argument(
        "--metadata",
        default="",
        help=(
            "Optional old JSON database used only to recover "
            "title/artist/url/video_id metadata"
        ),
    )
    parser.add_argument(
        "--original-suffix",
        default="",
        help=(
            "Only include WAV files ending with this suffix before .wav, "
            "for example _original"
        ),
    )
    return parser.parse_args()


def load_metadata(path: str) -> dict:
    if not path:
        candidates = [
            Path("song_fingerprints.before_noise_v2.json"),
            Path("song_fingerprints.pre_noise_v2_backup.json"),
            Path("song_fingerprints.old_chroma_backup.json"),
            Path("song_fingerprints.json"),
        ]
    else:
        candidates = [Path(path)]

    for candidate in candidates:
        if not candidate.exists():
            continue

        try:
            with candidate.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            continue

        if not isinstance(raw, dict):
            continue

        by_video_id = {}

        for title, data in raw.items():
            if not isinstance(data, dict):
                continue

            video_id = data.get("video_id")

            if not video_id:
                url = str(data.get("url", ""))
                match = re.search(r"[?&]v=([\w-]+)", url)
                if match:
                    video_id = match.group(1)

            if not video_id:
                continue

            by_video_id[str(video_id)] = {
                "title": title,
                "artist": data.get("artist", "Unknown"),
                "url": data.get("url", ""),
                "video_id": str(video_id),
            }

        if by_video_id:
            print(f"Metadata loaded from: {candidate}")
            return by_video_id

    return {}


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {
            "version": FINGERPRINT_VERSION,
            "songs": [],
        }

    try:
        from dsp_core import load_database
        return load_database(str(path))
    except Exception as exc:
        print(f"Existing database ignored: {exc}")
        return {
            "version": FINGERPRINT_VERSION,
            "songs": [],
        }


def file_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def choose_reference_files(
    folder: Path,
    original_suffix: str,
) -> list[Path]:
    files = sorted(folder.rglob("*.wav"))

    selected = []

    for path in files:
        stem_lower = path.stem.lower()

        if path.name.lower() == "last_recording.wav":
            continue

        if original_suffix:
            if not stem_lower.endswith(original_suffix.lower()):
                continue
        else:
            if any(stem_lower.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
                continue

        selected.append(path)

    return selected


def title_from_path(
    path: Path,
    original_suffix: str,
) -> str:
    title = path.stem

    if original_suffix and title.lower().endswith(
        original_suffix.lower()
    ):
        title = title[: -len(original_suffix)]

    return title.replace("_", " ").strip()


def main():
    args = parse_args()
    folder = Path(args.folder)
    database_path = Path(args.database)

    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    metadata = load_metadata(args.metadata)
    database = load_existing(database_path)

    existing_by_source = {
        str(song.get("source_path", "")): song
        for song in database.get("songs", [])
    }

    references = choose_reference_files(
        folder,
        args.original_suffix,
    )

    if not references:
        raise SystemExit(
            "No reference WAV files found. "
            "Check --folder and --original-suffix."
        )

    print(f"Reference WAV files found: {len(references)}")
    print(f"Existing cached tracks:    {len(database['songs'])}")
    print()

    new_count = 0
    cached_count = 0
    failed = []

    for index, wav_path in enumerate(references, 1):
        resolved = str(wav_path.resolve())
        size, mtime_ns = file_identity(wav_path)
        cached = existing_by_source.get(resolved)

        if (
            cached
            and cached.get("source_size") == size
            and cached.get("source_mtime_ns") == mtime_ns
            and cached.get("fingerprint_version")
            == FINGERPRINT_VERSION
        ):
            cached_count += 1
            print(
                f"[{index}/{len(references)}] Cached: "
                f"{cached['title']}"
            )
            continue

        print(
            f"[{index}/{len(references)}] Indexing: "
            f"{wav_path.name}"
        )

        try:
            sample_rate, audio = wavfile.read(wav_path)
            peaks, hashes = AudioProcessor.extract(
                audio,
                sample_rate,
                query_mode=False,
            )

            if len(peaks) < 50:
                raise RuntimeError(
                    f"too few peaks extracted: {len(peaks)}"
                )

            if len(hashes) < 100:
                raise RuntimeError(
                    f"too few landmark hashes extracted: {len(hashes)}"
                )

            video_id = wav_path.stem
            metadata_row = metadata.get(video_id, {})

            title = metadata_row.get("title") or title_from_path(
                wav_path,
                args.original_suffix,
            )

            song = {
                "title": title,
                "artist": metadata_row.get(
                    "artist",
                    "Unknown",
                ),
                "url": metadata_row.get("url", ""),
                "video_id": metadata_row.get(
                    "video_id",
                    video_id,
                ),
                "source_path": resolved,
                "source_size": size,
                "source_mtime_ns": mtime_ns,
                "fingerprint_version": FINGERPRINT_VERSION,
                "peaks": np.asarray(
                    peaks,
                    dtype=np.int32,
                ),
                "hashes": np.asarray(
                    hashes,
                    dtype=np.int32,
                ),
            }

            if cached:
                position = database["songs"].index(cached)
                database["songs"][position] = song
            else:
                database["songs"].append(song)

            existing_by_source[resolved] = song
            new_count += 1

            save_database(
                database,
                str(database_path),
            )

            print(
                f"    saved: peaks={len(peaks)}, "
                f"hashes={len(hashes)}"
            )

        except Exception as exc:
            failed.append((wav_path.name, str(exc)))
            print(f"    FAILED: {exc}")

    save_database(database, str(database_path))

    print()
    print(
        f"Done — {new_count} new/updated, "
        f"{cached_count} cached, "
        f"{len(database['songs'])} total songs."
    )
    print(f"Database: {database_path}")

    if failed:
        print(f"\n{len(failed)} files failed:")
        for filename, reason in failed:
            print(f"  - {filename}: {reason}")


if __name__ == "__main__":
    main()
