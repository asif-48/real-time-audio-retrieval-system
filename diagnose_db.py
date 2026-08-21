"""
diagnose_db.py — Inspect the real, indexed database for structural quirks
========================================================================
Runs no queries — just reports, per track, what fraction of its frames are
silent (per the same energy floor extract_fingerprint uses) and how
"static" its content is on average. If one or two tracks (e.g. the ones
that keep winning regardless of query) stand out with much higher silent
fractions or much lower frame-to-frame variance than the rest, that's the
real, data-backed confirmation of the sustained/quiet-passage hypothesis —
instead of guessing further from synthetic tones.

Run:
    python diagnose_db.py
"""

import json
import numpy as np

from dsp_core import DB_FILE


def analyze(fp: np.ndarray):
    chroma = np.array(fp)[:, :12]   # first 12 dims are chroma; last 12 are delta
    silent_frac = float(np.mean(np.all(chroma == 0, axis=1)))
    # Average frame-to-frame change among the NON-silent frames -- a proxy
    # for "how static/sustained is this track's melodic content on average".
    active = chroma[~np.all(chroma == 0, axis=1)]
    if len(active) > 1:
        frame_deltas = np.linalg.norm(np.diff(active, axis=0), axis=1)
        avg_movement = float(np.mean(frame_deltas))
    else:
        avg_movement = float('nan')
    return silent_frac, avg_movement


def main():
    with open(DB_FILE) as f:
        db = json.load(f)

    rows = []
    for title, data in db.items():
        fp = data.get("fingerprint")
        if not fp:
            continue
        silent_frac, avg_movement = analyze(fp)
        rows.append((title, silent_frac, avg_movement, len(fp)))

    # Sort by silent fraction, most-silent first -- these are the tracks
    # most likely to be exploitable "attractors" for unrelated queries.
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"{'Track':<55} {'silent%':>8} {'avg_move':>9} {'frames':>7}")
    print("-" * 82)
    for title, silent_frac, avg_movement, n_frames in rows:
        short_title = (title[:52] + "...") if len(title) > 55 else title
        print(f"{short_title:<55} {silent_frac*100:>7.1f}% {avg_movement:>9.4f} {n_frames:>7}")

    print()
    silents = [r[1] for r in rows]
    valid_movement_rows = [r for r in rows if not np.isnan(r[2])]
    print(f"Silent%  median={np.median(silents)*100:.1f}%  max={max(silents)*100:.1f}% ({rows[0][0][:40]})")
    if valid_movement_rows:
        min_row = min(valid_movement_rows, key=lambda r: r[2])
        movements = [r[2] for r in valid_movement_rows]
        print(f"avg_move median={np.median(movements):.4f}  min={min_row[2]:.4f} ({min_row[0][:40]})")
    print()
    print("Look for: tracks with silent% or avg_move far outside the pack --")
    print("those are the ones most likely to act as 'attractor' false matches")
    print("regardless of what's actually queried.")


if __name__ == "__main__":
    main()
