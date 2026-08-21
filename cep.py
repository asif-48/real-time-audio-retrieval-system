
"""
Closed-set live detector for a fixed song library.

The program never indexes songs and never accesses the network.
Build the database once with build_database.py, then run this file normally.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import tkinter as tk
from scipy.io import wavfile
from tkinter import messagebox, ttk

from dsp_core import (
    AudioProcessor,
    ClosedSetMatcher,
    DB_FILE,
    LOG_FILE,
    MIN_QUERY_HASHES,
    MIN_QUERY_PEAKS,
    RECORD_SECONDS,
    load_database,
    log_match,
    log_no_match,
)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError) as exc:
    SOUNDDEVICE_AVAILABLE = False
    SOUNDDEVICE_ERROR = str(exc)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

METHOD_NAME = "Closed-set landmark and peak voting"


class MusicRecognitionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("52-Song Offline Detector")
        self.root.geometry("820x610")
        self.root.configure(bg="#263746")
        self.root.resizable(True, True)

        self.database = None
        self.matcher = None
        self.is_recording = False
        self.ui_queue = queue.Queue()
        self.input_devices = []

        self._load_database()
        self._build_ui()
        self._refresh_devices()
        self._poll_queue()

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    def _load_database(self):
        try:
            self.database = load_database(DB_FILE)
            self.matcher = ClosedSetMatcher(self.database)
            self.database_error = ""
        except Exception as exc:
            self.database = None
            self.matcher = None
            self.database_error = str(exc)

    def _reload_database(self):
        self._load_database()

        if self.database:
            count = len(self.database["songs"])
            self.db_label.config(
                text=f"Database: {count} songs",
                fg="#2ecc71",
            )
            self.status_label.config(
                text=f"Database reloaded. {count} songs ready.",
                fg="#2ecc71",
            )
        else:
            self.db_label.config(
                text="Database unavailable",
                fg="#e74c3c",
            )
            self.status_label.config(
                text=(
                    "Run build_database.py once. "
                    f"{self.database_error}"
                ),
                fg="#e74c3c",
            )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        header = tk.Frame(
            self.root,
            bg="#34495e",
            padx=12,
            pady=10,
        )
        header.pack(
            fill=tk.X,
            padx=12,
            pady=(12, 5),
        )

        tk.Label(
            header,
            text="Offline Closed-Set Song Detector",
            bg="#34495e",
            fg="white",
            font=("Arial", 13, "bold"),
        ).pack(side=tk.LEFT)

        count = (
            len(self.database["songs"])
            if self.database
            else 0
        )

        self.db_label = tk.Label(
            header,
            text=(
                f"Database: {count} songs"
                if count
                else "Database unavailable"
            ),
            bg="#34495e",
            fg="#2ecc71" if count else "#e74c3c",
            font=("Arial", 10, "bold"),
        )
        self.db_label.pack(side=tk.RIGHT)

        controls = tk.LabelFrame(
            self.root,
            text="  Input and Recording  ",
            bg="#263746",
            fg="#f1c40f",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=10,
        )
        controls.pack(
            fill=tk.X,
            padx=18,
            pady=6,
        )

        device_row = tk.Frame(
            controls,
            bg="#263746",
        )
        device_row.pack(fill=tk.X)

        tk.Label(
            device_row,
            text="Input device:",
            bg="#263746",
            fg="#ecf0f1",
            font=("Arial", 9),
        ).pack(side=tk.LEFT)

        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(
            device_row,
            textvariable=self.device_var,
            state="readonly",
            width=58,
        )
        self.device_box.pack(
            side=tk.LEFT,
            padx=8,
            fill=tk.X,
            expand=True,
        )

        tk.Button(
            device_row,
            text="Refresh",
            command=self._refresh_devices,
            bg="#7f8c8d",
            fg="white",
            relief=tk.FLAT,
            padx=8,
        ).pack(side=tk.RIGHT)

        button_row = tk.Frame(
            controls,
            bg="#263746",
        )
        button_row.pack(pady=(12, 0))

        self.record_button = tk.Button(
            button_row,
            text=(
                f"🎙  LISTEN {RECORD_SECONDS}s "
                "AND IDENTIFY"
            ),
            command=self._start_recording,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 13, "bold"),
            width=30,
            height=2,
            relief=tk.FLAT,
        )
        self.record_button.pack(side=tk.LEFT, padx=8)

        tk.Button(
            button_row,
            text="Reload DB",
            command=self._reload_database,
            bg="#7f8c8d",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12,
            height=2,
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=8)

        self.status_label = tk.Label(
            self.root,
            text=(
                "Ready. Play one indexed song through the phone speaker."
                if self.database
                else (
                    "Run build_database.py once. "
                    f"{self.database_error}"
                )
            ),
            bg="#263746",
            fg="#aab7c4" if self.database else "#e74c3c",
            font=("Arial", 10, "italic"),
            wraplength=760,
        )
        self.status_label.pack(pady=8)

        result_frame = tk.LabelFrame(
            self.root,
            text="  Detected Song  ",
            bg="#263746",
            fg="#f1c40f",
            font=("Arial", 11, "bold"),
            padx=12,
            pady=10,
        )
        result_frame.pack(
            fill=tk.X,
            padx=18,
            pady=5,
        )

        self.title_label = tk.Label(
            result_frame,
            text="Song: —",
            bg="#263746",
            fg="white",
            font=("Arial", 16, "bold"),
            anchor="w",
        )
        self.title_label.pack(fill=tk.X)

        self.artist_label = tk.Label(
            result_frame,
            text="Artist / Source: —",
            bg="#263746",
            fg="#bdc3c7",
            font=("Arial", 10),
            anchor="w",
        )
        self.artist_label.pack(fill=tk.X, pady=(4, 0))

        self.evidence_label = tk.Label(
            result_frame,
            text="Evidence: —",
            bg="#263746",
            fg="#3498db",
            font=("Arial", 10, "bold"),
            anchor="w",
            justify=tk.LEFT,
        )
        self.evidence_label.pack(fill=tk.X, pady=(5, 0))

        candidates_frame = tk.LabelFrame(
            self.root,
            text="  Ranking Details  ",
            bg="#263746",
            fg="#f1c40f",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=8,
        )
        candidates_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=18,
            pady=(5, 14),
        )

        self.candidate_box = tk.Text(
            candidates_frame,
            bg="#1c2a35",
            fg="#ecf0f1",
            font=("Consolas", 10),
            height=10,
            state=tk.DISABLED,
            relief=tk.FLAT,
        )
        self.candidate_box.pack(
            fill=tk.BOTH,
            expand=True,
        )

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------
    def _refresh_devices(self):
        self.input_devices = []

        if not SOUNDDEVICE_AVAILABLE:
            self.device_box["values"] = []
            self.device_var.set("sounddevice unavailable")
            return

        try:
            devices = sd.query_devices()
            default_input = (
                sd.default.device[0]
                if isinstance(sd.default.device, (tuple, list))
                else sd.default.device
            )

            values = []
            selected_position = 0

            for device_index, device in enumerate(devices):
                if int(device.get("max_input_channels", 0)) <= 0:
                    continue

                label = (
                    f"{device_index}: {device['name']} "
                    f"({int(device['default_samplerate'])} Hz)"
                )
                self.input_devices.append(
                    (device_index, device)
                )
                values.append(label)

                if device_index == default_input:
                    selected_position = len(values) - 1

            self.device_box["values"] = values

            if values:
                self.device_box.current(selected_position)
            else:
                self.device_var.set("No input devices found")

        except Exception as exc:
            self.device_box["values"] = []
            self.device_var.set(f"Device error: {exc}")

    def _selected_device(self):
        position = self.device_box.current()

        if position < 0 or position >= len(self.input_devices):
            raise RuntimeError("Select a valid microphone input device")

        return self.input_devices[position]

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                function, args, kwargs = self.ui_queue.get_nowait()
                function(*args, **kwargs)
        except queue.Empty:
            pass

        self.root.after(120, self._poll_queue)

    def _post_ui(self, function, *args, **kwargs):
        self.ui_queue.put((function, args, kwargs))

    # ------------------------------------------------------------------
    # Recording and matching
    # ------------------------------------------------------------------
    def _start_recording(self):
        if not self.database or not self.matcher:
            messagebox.showwarning(
                "Database missing",
                "Run build_database.py once, then click Reload DB.",
            )
            return

        if not SOUNDDEVICE_AVAILABLE:
            messagebox.showerror(
                "Microphone unavailable",
                (
                    "sounddevice could not load:\n"
                    f"{SOUNDDEVICE_ERROR}"
                ),
            )
            return

        if self.is_recording:
            return

        try:
            device_index, device = self._selected_device()
        except Exception as exc:
            messagebox.showerror("Input device", str(exc))
            return

        self.is_recording = True
        self.record_button.config(
            state=tk.DISABLED,
            bg="#95a5a6",
        )

        threading.Thread(
            target=self._recording_worker,
            args=(device_index, device),
            daemon=True,
        ).start()

    def _restore_button(self):
        self.is_recording = False
        self._post_ui(
            self.record_button.config,
            state=tk.NORMAL,
            text=(
                f"🎙  LISTEN {RECORD_SECONDS}s "
                "AND IDENTIFY"
            ),
            bg="#2ecc71",
        )

    def _recording_worker(self, device_index, device):
        try:
            record_rate = int(
                round(float(device["default_samplerate"]))
            )

            if record_rate < 8000:
                record_rate = 44100

            sample_count = int(RECORD_SECONDS * record_rate)

            logging.info(
                "RECORDING | device_index=%d | device=%r | "
                "sample_rate=%d | seconds=%d",
                device_index,
                device.get("name", "Unknown"),
                record_rate,
                RECORD_SECONDS,
            )

            recording = sd.rec(
                sample_count,
                samplerate=record_rate,
                channels=1,
                dtype="float32",
                device=device_index,
            )

            for remaining in range(RECORD_SECONDS, 0, -1):
                self._post_ui(
                    self.record_button.config,
                    text=f"🎙  Recording… {remaining}s",
                    bg="#e74c3c",
                )
                time.sleep(1.0)

            sd.wait()
            audio = np.asarray(
                recording[:, 0],
                dtype=np.float32,
            )

        except Exception as exc:
            logging.error(
                "Recording failed: %s",
                exc,
                exc_info=True,
            )
            self._post_ui(
                messagebox.showerror,
                "Recording Error",
                str(exc),
            )
            self._post_ui(
                self.status_label.config,
                text="Recording failed. Check the selected microphone.",
                fg="#e74c3c",
            )
            self._restore_button()
            return

        peak_level = (
            float(np.max(np.abs(audio)))
            if audio.size
            else 0.0
        )
        rms_level = (
            float(np.sqrt(np.mean(audio ** 2)))
            if audio.size
            else 0.0
        )
        clipped_fraction = (
            float(np.mean(np.abs(audio) >= 0.98))
            if audio.size
            else 0.0
        )

        logging.info(
            "RECORDING_LEVEL | peak=%.5f | rms=%.5f | "
            "clipped=%.3f%%",
            peak_level,
            rms_level,
            100.0 * clipped_fraction,
        )

        try:
            wavfile.write(
                "last_recording.wav",
                record_rate,
                audio,
            )
        except Exception as exc:
            logging.warning(
                "Could not save last_recording.wav: %s",
                exc,
            )

        if peak_level < 0.006 or rms_level < 0.0006:
            log_no_match("recording too quiet")
            self._post_ui(
                self.status_label.config,
                text=(
                    "The microphone recording is too quiet. "
                    "Play last_recording.wav and verify the selected input."
                ),
                fg="#e74c3c",
            )
            self._post_ui(self._show_no_result)
            self._restore_button()
            return

        self._post_ui(
            self.status_label.config,
            text="Extracting fingerprints and ranking all songs…",
            fg="#e67e22",
        )

        try:
            query_peaks, query_hashes = AudioProcessor.extract(
                audio,
                record_rate,
                query_mode=True,
            )

            logging.info(
                "QUERY | peaks=%d | hashes=%d",
                len(query_peaks),
                len(query_hashes),
            )

            if len(query_peaks) < MIN_QUERY_PEAKS:
                raise RuntimeError(
                    f"Too few stable peaks: {len(query_peaks)}"
                )

            if len(query_hashes) < MIN_QUERY_HASHES:
                raise RuntimeError(
                    f"Too few landmark hashes: {len(query_hashes)}"
                )

            best, alternatives = self.matcher.match(
                query_peaks,
                query_hashes,
            )

            if best is None:
                raise RuntimeError(
                    "No usable match evidence was produced"
                )

            self._post_ui(
                self._show_result,
                best,
                alternatives,
            )

            warning = ""

            if clipped_fraction > 0.05:
                warning = (
                    " Recording clipped heavily; lower the phone "
                    "or microphone level if results become unstable."
                )

            self._post_ui(
                self.status_label.config,
                text=(
                    "Finished. The strongest song in the fixed "
                    f"library was selected.{warning}"
                ),
                fg="#2ecc71" if not warning else "#f1c40f",
            )

        except Exception as exc:
            logging.error(
                "Matching failed: %s",
                exc,
                exc_info=True,
            )
            self._post_ui(
                self.status_label.config,
                text=(
                    f"Could not classify this recording: {exc}. "
                    "Check last_recording.wav."
                ),
                fg="#e74c3c",
            )
            self._post_ui(self._show_no_result)

        finally:
            self._restore_button()

    # ------------------------------------------------------------------
    # Result rendering
    # ------------------------------------------------------------------
    def _show_no_result(self):
        self.title_label.config(
            text="Song: No usable audio",
            fg="#e74c3c",
        )
        self.artist_label.config(
            text="Artist / Source: —"
        )
        self.evidence_label.config(
            text="Evidence: —"
        )
        self.candidate_box.config(state=tk.NORMAL)
        self.candidate_box.delete("1.0", tk.END)
        self.candidate_box.insert(
            tk.END,
            "No ranking was produced.\n",
        )
        self.candidate_box.config(state=tk.DISABLED)

    def _show_result(self, best, alternatives):
        self.title_label.config(
            text=f"Song: {best['title']}",
            fg="#2ecc71",
        )
        self.artist_label.config(
            text=(
                "Artist / Source: "
                f"{best.get('artist', 'Unknown')}"
            )
        )
        self.evidence_label.config(
            text=(
                f"Evidence score: {best['confidence']:.1f}%  |  "
                f"winner/runner-up ratio: "
                f"{best['winner_ratio']:.2f}×  |  "
                f"aligned landmark votes: "
                f"{best['hash_votes']:.0f}  |  "
                f"aligned peak votes: "
                f"{best['peak_votes']:.1f}"
            )
        )

        log_match(
            best["title"],
            best["confidence"],
            METHOD_NAME,
        )

        rows = [best] + list(alternatives or [])

        self.candidate_box.config(state=tk.NORMAL)
        self.candidate_box.delete("1.0", tk.END)

        self.candidate_box.insert(
            tk.END,
            (
                "Rank  Song"
                "                               "
                "Hash votes  Peak votes  Score\n"
            ),
        )
        self.candidate_box.insert(
            tk.END,
            "-" * 75 + "\n",
        )

        for rank, row in enumerate(rows[:5], 1):
            title = row["title"]
            if len(title) > 34:
                title = title[:31] + "..."

            self.candidate_box.insert(
                tk.END,
                (
                    f"{rank:<5} "
                    f"{title:<35} "
                    f"{row['hash_votes']:>10.0f} "
                    f"{row['peak_votes']:>11.1f} "
                    f"{row['ranking_score']:>8.1f}\n"
                ),
            )

        self.candidate_box.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicRecognitionApp(root)
    root.mainloop()
