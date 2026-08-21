"""
reindex.py — Re-extract fingerprints from already-downloaded audio
========================================================================
Use this instead of re-running build_database.py whenever dsp_core.py's
feature-extraction logic changes (like the silence-masking fix) but the
actual song audio hasn't. It:
  1. Loads the existing song_fingerprints.json (for title/artist/url —
     no need to re-fetch playlist metadata).
  2. Recovers each track's video ID from its stored url.
  3. Re-extracts the fingerprint from the WAV already sitting in
     downloaded_tracks/, using the CURRENT AudioProcessor.
  4. Saves the updated song_fingerprints.json.

No yt_dlp import, no network access — this can't hit a YouTube block.

Run:
    python reindex.py
"""

import os
import re
import json
import logging

import numpy as np
from scipy.io import wavfile
import scipy.signal as signal

from dsp_core import AudioProcessor, SAMPLE_RATE, FRAME_SIZE, DB_FILE, LOG_FILE

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

STORAGE_DIR = "downloaded_tracks"


def video_id_from_url(url: str) -> str:
    match = re.search(r"[?&]v=([\w-]+)", url or "")
    return match.group(1) if match else None


def main():
    if not os.path.exists(DB_FILE):
        print(f"ERROR: {DB_FILE} not found. Run build_database.py first (one time, needs network).")
        return

    with open(DB_FILE) as f:
        old_db = json.load(f)

    print(f"Re-fingerprinting {len(old_db)} tracks from local audio in {STORAGE_DIR}/ (no network)…")

    new_db = {}
    skipped = []

    for i, (title, data) in enumerate(old_db.items(), 1):
        url = data.get("url", "")
        vid = video_id_from_url(url)
        if not vid:
            skipped.append((title, f"couldn't recover video ID from url: {url}"))
            continue

        wav_path = os.path.join(STORAGE_DIR, f"{vid}.wav")
        if not os.path.exists(wav_path):
            skipped.append((title, f"no local WAV at {wav_path} — run build_database.py to fetch it"))
            continue

        print(f"[{i}/{len(old_db)}] Re-fingerprinting: {title[:60]}")
        try:
            fs, audio = wavfile.read(wav_path)
            if audio is None or audio.size == 0:
                skipped.append((title, "empty audio file"))
                continue
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = audio.astype(np.float32)
            if audio.size < FRAME_SIZE:
                skipped.append((title, "audio too short"))
                continue

            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio /= max_val
            if fs != SAMPLE_RATE:
                n_samples = int(len(audio) * SAMPLE_RATE / fs)
                audio = signal.resample(audio, n_samples)

            fp = AudioProcessor.extract_fingerprint(audio, SAMPLE_RATE)
            new_db[title] = {
                "artist": data.get("artist", "Unknown"),
                "url": url,
                "fingerprint": fp.tolist(),
            }
            logging.info(f"Reindexed: '{title}' | id={vid} | frames={fp.shape[0]}")

        except Exception as e:
            skipped.append((title, f"fingerprint failed: {e}"))
            logging.error(f"Reindex failed for '{title}': {e}", exc_info=True)

    with open(DB_FILE, "w") as f:
        json.dump(new_db, f, indent=2)

    print(f"\nDone: {len(new_db)}/{len(old_db)} tracks re-fingerprinted into {DB_FILE}")
    if skipped:
        print(f"{len(skipped)} skipped:")
        for t, reason in skipped:
            print(f"  - {t}: {reason}")


if __name__ == "__main__":
    main()
