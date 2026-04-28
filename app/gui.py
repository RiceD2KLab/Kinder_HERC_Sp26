"""
gui.py — Main GUI entry point for the YouTube Transcript Highlighter app.
Wires together: yt-dlp download → Parakeet transcription → chunker → LR model → .docx output
Supports two input modes:
  • YouTube URL  → download audio → transcribe → chunk → predict
  • Local audio file → transcribe → chunk → predict  (skips download step)
"""

import customtkinter as ctk
import threading
import tempfile
import os
import sys
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

#to help w path finding
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

#hardcoding ffmpeg location
_BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
FFMPEG_PATH = _BIN_DIR / "ffmpeg.exe"

#adding this for Nemo not finding ffmpeg
if _BIN_DIR.exists():
    os.environ["PATH"] += os.pathsep + str(_BIN_DIR)

from pydub import AudioSegment
AudioSegment.converter = str(FFMPEG_PATH)
AudioSegment.ffprobe = str(_BIN_DIR / "ffprobe.exe")


# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

AUDIO_EXTENSIONS = (
    ".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"
)
def _collect_audio_files(paths: list) -> list:
    found = []
    for p in paths:
        if p.is_dir():
            found.extend(
                sorted(f for f in p.rglob("*")
                       if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS)
            )
        elif p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            found.append(p)
    return found
 
 
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HERC Research Mention Finder")
        self.geometry("700x780")
        self.resizable(False, False)
 
        self._result_dir = None
        self._selected_audio_paths = []
 
        self._build_ui()
 
    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        ctk.CTkLabel(
            self, text="HERC Research Mention Finder",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(24, 2))
 
        ctk.CTkLabel(
            self,
            text="This tool can be used to find research mentions in school board meetings. \n Provide a URL to a school board meeting online or upload the video file directly to begin the pipeline.",
            font=ctk.CTkFont(size=13), text_color="gray"
        ).pack(pady=(0, 16))
 
        # ── Tab view ───────────────────────────────────────────────────────────
        self.tab_view = ctk.CTkTabview(self, height=200)
        self.tab_view.pack(fill="x", padx=40)
 
        self.tab_view.add("Website URL")
        self.tab_view.add("Upload File(s)")
 
        # Tab 1 — YouTube URL
        tab_url = self.tab_view.tab("Website URL")
        ctk.CTkLabel(
            tab_url, text="Website URL",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w")
        self.url_entry = ctk.CTkEntry(
            tab_url,
            placeholder_text="https://www.swagit.com/...",
            height=40, font=ctk.CTkFont(size=13)
        )
        self.url_entry.pack(fill="x", pady=(4, 0))
 
        # ctk.CTkLabel(
        #     tab_url, text="ffmpeg location  (optional — leave blank if ffmpeg is on PATH)",
        #     font=ctk.CTkFont(size=11), text_color="gray"
        # ).pack(anchor="w", pady=(10, 0))
 
        # ffmpeg_row = ctk.CTkFrame(tab_url, fg_color="transparent")
        # ffmpeg_row.pack(fill="x", pady=(4, 0))
        # self.ffmpeg_entry = ctk.CTkEntry(
        #     ffmpeg_row,
        #     placeholder_text="e.g.  C:/ffmpeg/bin/ffmpeg.exe  or  /usr/local/bin/ffmpeg",
        #     height=36, font=ctk.CTkFont(size=12)
        # )
        # self.ffmpeg_entry.pack(side="left", fill="x", expand=True)
        # ctk.CTkButton(
        #     ffmpeg_row, text="Browse", width=80, height=36,
        #     command=self._pick_ffmpeg_path
        # ).pack(side="left", padx=(8, 0))
 
        # Tab 2 — Audio File(s) / Folder
        tab_audio = self.tab_view.tab("Upload File(s)")
        ctk.CTkLabel(
            tab_audio,
            text="Select one or more audio files, or a folder containing audio files.",
            font=ctk.CTkFont(size=12), text_color="gray"
        ).pack(anchor="w", pady=(0, 6))
 
        btn_row = ctk.CTkFrame(tab_audio, fg_color="transparent")
        btn_row.pack(fill="x")
        ctk.CTkButton(
            btn_row, text="Select File(s)", width=130, height=36,
            command=self._pick_audio_files
        ).pack(side="left")
        ctk.CTkButton(
            btn_row, text="Select Folder", width=130, height=36,
            command=self._pick_audio_folder
        ).pack(side="left", padx=(10, 0))
        ctk.CTkButton(
            btn_row, text="Clear", width=70, height=36,
            fg_color="transparent", border_width=1,
            text_color=("gray40", "gray60"),
            hover_color=("gray85", "gray25"),
            command=self._clear_audio_selection
        ).pack(side="left", padx=(10, 0))
 
        self.audio_summary_label = ctk.CTkLabel(
            tab_audio, text="No files selected.",
            font=ctk.CTkFont(size=12), text_color="gray", anchor="w"
        )
        self.audio_summary_label.pack(fill="x", pady=(8, 0))
 
        # ── Output directory ───────────────────────────────────────────────────
        out_frame = ctk.CTkFrame(self, fg_color="transparent")
        out_frame.pack(fill="x", padx=40, pady=(14, 0))
        ctk.CTkLabel(
            out_frame, text="Save output to",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w")
        dir_row = ctk.CTkFrame(out_frame, fg_color="transparent")
        dir_row.pack(fill="x", pady=(4, 0))
        self.out_dir_var = tk.StringVar(value=str(Path.home() / "Downloads"))
        ctk.CTkEntry(
            dir_row, textvariable=self.out_dir_var,
            height=40, font=ctk.CTkFont(size=12)
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            dir_row, text="Browse", width=80, height=40,
            command=self._pick_output_dir
        ).pack(side="left", padx=(8, 0))
 
        # ── District name ──────────────────────────────────────────────────────
        district_frame = ctk.CTkFrame(self, fg_color="transparent")
        district_frame.pack(fill="x", padx=40, pady=(14, 0))
        ctk.CTkLabel(
            district_frame, text="District name",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w")
        self.district_entry = ctk.CTkEntry(
            district_frame, placeholder_text="e.g. Katy ISD",
            height=40, font=ctk.CTkFont(size=13)
        )
        self.district_entry.pack(fill="x", pady=(4, 0))
 
        # ── Run button ─────────────────────────────────────────────────────────
        self.run_btn = ctk.CTkButton(
            self, text="Run Pipeline", height=46,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start_pipeline
        )
        self.run_btn.pack(padx=40, pady=(20, 0), fill="x")
 
        # ── Retry button (hidden until an error occurs) ────────────────────────
        self.retry_btn = ctk.CTkButton(
            self, text="↺  Retry", height=38,
            font=ctk.CTkFont(size=13),
            fg_color="#8B3A3A", hover_color="#6b2a2a",
            command=self._retry
        )
        # Packed but invisible until needed
        self.retry_btn.pack(padx=40, pady=(6, 0), fill="x")
        self.retry_btn.pack_forget()
 
        # ── Progress ───────────────────────────────────────────────────────────
        prog_frame = ctk.CTkFrame(self, fg_color="transparent")
        prog_frame.pack(fill="x", padx=40, pady=(12, 0))
        self.progress = ctk.CTkProgressBar(prog_frame, height=10)
        self.progress.set(0)
        self.progress.pack(fill="x")
        self.status_label = ctk.CTkLabel(
            prog_frame, text="Ready.",
            font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.status_label.pack(anchor="w", pady=(4, 0))
 
        # ── Log box ────────────────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self, fg_color="transparent")
        log_frame.pack(fill="both", expand=True, padx=40, pady=(12, 0))
        ctk.CTkLabel(
            log_frame, text="Log",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w")
        self.log_box = ctk.CTkTextbox(
            log_frame, height=160,
            font=ctk.CTkFont(family="Courier", size=11),
            state="disabled"
        )
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))
 
        # ── Open output folder button ──────────────────────────────────────────
        self.open_btn = ctk.CTkButton(
            self, text="📂  Open Output Folder",
            height=42, font=ctk.CTkFont(size=13),
            fg_color="green", hover_color="#2d7a2d",
            command=self._open_output_folder, state="disabled"
        )
        self.open_btn.pack(padx=40, pady=(12, 22), fill="x")
 
    # ── File pickers ───────────────────────────────────────────────────────────
    # def _pick_ffmpeg_path(self):
    #     path = filedialog.askopenfilename(
    #         title="Locate ffmpeg binary",
    #         filetypes=[("Executable", "ffmpeg ffmpeg.exe *"), ("All files", "*.*")],
    #     )
    #     if path:
    #         self.ffmpeg_entry.delete(0, "end")
    #         self.ffmpeg_entry.insert(0, path)
 
    def _pick_audio_files(self):
        filetypes = [
            ("Upload files", " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)),
            ("All files", "*.*"),
        ]
        paths = filedialog.askopenfilenames(
            title="Select audio file(s)", filetypes=filetypes
        )
        if paths:
            self._selected_audio_paths = [Path(p) for p in paths]
            self._update_audio_summary()
 
    def _pick_audio_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing audio files")
        if folder:
            found = _collect_audio_files([Path(folder)])
            if not found:
                messagebox.showwarning(
                    "No audio files found",
                    f"No supported audio files were found in:\n{folder}\n\n"
                    f"Supported: {', '.join(AUDIO_EXTENSIONS)}"
                )
                return
            self._selected_audio_paths = found
            self._update_audio_summary()
 
    def _clear_audio_selection(self):
        self._selected_audio_paths = []
        self.audio_summary_label.configure(text="No files selected.")
 
    def _update_audio_summary(self):
        n = len(self._selected_audio_paths)
        if n == 0:
            self.audio_summary_label.configure(text="No files selected.")
        elif n == 1:
            self.audio_summary_label.configure(
                text=f"1 file: {self._selected_audio_paths[0].name}"
            )
        else:
            self.audio_summary_label.configure(
                text=f"{n} files queued  (e.g. {self._selected_audio_paths[0].name}, …)"
            )
 
    def _pick_output_dir(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.out_dir_var.set(d)
 
    # ── Logging / status ───────────────────────────────────────────────────────
    def _log(self, msg):
        """Safe to call from any thread via self.after(0, self._log, msg)."""
        self.log_box.configure(state="normal")
        self.log_box.insert("end", str(msg) + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
 
    def _set_status(self, msg: str, progress: float):
        self.status_label.configure(text=msg)
        self.progress.set(progress)
 
    def _open_output_folder(self):
        if self._result_dir and Path(self._result_dir).exists():
            import subprocess, platform
            if platform.system() == "Darwin":
                subprocess.call(["open", str(self._result_dir)])
            elif platform.system() == "Windows":
                os.startfile(str(self._result_dir))
            else:
                subprocess.call(["xdg-open", str(self._result_dir)])
 
    # ── Retry ──────────────────────────────────────────────────────────────────
    def _retry(self):
        """Hide retry button, clear the log, reset progress, and run again."""
        self.retry_btn.pack_forget()
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.progress.set(0)
        self.status_label.configure(text="Retrying…", text_color="gray")
        self._start_pipeline()
 
    def _show_error(self, short_msg: str, full_trace: str):
        """Called on the main thread when the pipeline fails."""
        self._set_status(f" {short_msg}", 0)
        self._log(f"\n {short_msg}\n{full_trace}")
        # Re-enable run button and show retry button
        self.run_btn.configure(state="normal")
        self.retry_btn.pack(padx=40, pady=(6, 0), fill="x",
                            before=self.progress.master)
 
    # ── Validation + dispatch ──────────────────────────────────────────────────
    def _start_pipeline(self):
        active_tab = self.tab_view.get()
 
        if active_tab == "Website URL":
            url = self.url_entry.get().strip()
            if not url:
                messagebox.showwarning("Missing URL", "Please enter a valid URL.")
                return
            audio_paths = None
        else:
            if not self._selected_audio_paths:
                messagebox.showwarning(
                    "No files selected",
                    "Please select at least one audio file or a folder."
                )
                return
            audio_paths = self._selected_audio_paths
            url = None
 
        district = self.district_entry.get().strip()
        if not district:
            messagebox.showwarning(
                "Missing district", "Please enter a district name (e.g. Katy ISD)."
            )
            return
 
        out_dir = Path(self.out_dir_var.get().strip())
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Bad output path", str(e))
            return
 
        ffmpeg_location = FFMPEG_PATH
 
        self.run_btn.configure(state="disabled")
        self.retry_btn.pack_forget()
        self.open_btn.configure(state="disabled")
        self._result_dir = None
 
        threading.Thread(
            target=self._run_pipeline,
            args=(url, audio_paths, out_dir, district, ffmpeg_location),
            daemon=True
        ).start()
   



    def _run_pipeline(self, url: str | None, audio_paths, out_dir: Path, district: str, ffmpeg_location):
        try:
            with tempfile.TemporaryDirectory(prefix="transcript_") as tmp:
                tmp_path = Path(tmp)

                # ── Step 1: Get audio ──────────────────────────────────────
                if url:
                    # Mode A: download from YouTube
                    self.after(0, self._set_status, "Step 1/4 — Downloading audio…", 0.10)
                    self.after(0, self._log, f"[1/4] Downloading: {url}")

                    scrape_dir = tmp_path / "scraped"
                    scrape_dir.mkdir()

                    from web_scraping.cli import run_scraping_pipeline
                    from web_scraping.models import Source

                    #Create Source object
                    user_source = Source(district=district, url=url)
                    #run function to download and convert to WAV
                    run_scraping_pipeline(
                        frag_workers=8,
                        max_candidates=1,
                        include_title=".*",
                        include_anchor_label=".*",
                        workers=8,
                        sources=[user_source],
                        out_path=scrape_dir,
                        cutoff_str= "2024-09-01",
                        ffmpeg_loc=Path(ffmpeg_location) if ffmpeg_location else None)
                    
                    self.after(0, self._log, "Audio saved")

                    audio_files = sorted(
                        f for f in scrape_dir.rglob("*")
                        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
                    )
 
                    if not audio_files:
                        raise FileNotFoundError(
                            f"Scraper finished but no audio files found in {scrape_dir}\n"
                            "Check that the URL is accessible and ffmpeg is configured."
                        )
                    
                    #Printing sanity checks
                    self.after(0, self._log,
                               f"      Found {len(audio_files)} audio file(s):")
                    for f in audio_files:
                        self.after(0, self._log, f"      {f.name}")


                else:
                    # Mode B: use the local file directly — skip download
                    audio_files = audio_paths
                    source_labels = {p: str(p) for p in audio_files}
                    self.after(0, self._set_status,
                               f"Step 1/4 — {len(audio_files)} file(s) queued.", 0.05)
                    self.after(0, self._log,
                               f"[1/4] {len(audio_files)} local file(s) ready:")
                    for p in audio_files:
                        self.after(0, self._log, f"      {p.name}")
 
                n = len(audio_files)

                # ── Step 2: Transcribe ─────────────────────────────────────
                self.after(0, self._set_status, f"Step 2/4 — {n} file(s)…", 0.30)
                self.after(0, self._log, f"[2/4] Transcribing with Parakeet {n} file(s)…")

                transcript_dir = tmp_path / "transcripts"
                transcript_dir.mkdir()

                from transcription.parakeet_transcribe import run_transcription, load_asr_model
                self.after(0, self._log,"Attempting to load ASR model")
                asr_model = load_asr_model()
                self.after(0, self._log, "Model successfully loaded. Beginning transcription...")

                for i, ap in enumerate(audio_files, 1):
                    self.after(0, self._log, f"      [{i}/{n}] {ap.name}")
                    run_transcription(
                        audio_path=ap,              # ← input: full path to audio
                        output_dir=transcript_dir,  # ← output: where .txt is written
                        asr_model=asr_model,
                        ffmpeg_path=ffmpeg_location or None
                    )

                self.after(0, self._set_status,
                               f"Step 2/4 — Transcribed {i}/{n}")



                # audio_path = transcript_dir / audio_files[0]
                # self.after(0, self._log, f"      Transcript: {audio_path.name}")

                # transcript_path = run_transcription(audio_path, transcript_dir)
                # self._log(transcript_path)

                # ── Step 3: Chunk ──────────────────────────────────────────
                self.after(0, self._set_status, f"Step 3/4 — Chunking {n} transcript…", 0.60)
                self.after(0, self._log, "[3/4] Splitting into 2-minute chunks…")

                from transcript_chunking import create_chunks           
                chunks_dir = tmp_path/"chunks"
                chunks_dir.mkdir()
                
                chunks_map = {}
 
                for i, ap in enumerate(audio_files, 1):
                    # The transcript was written to transcript_dir/<audio_stem>.txt
                    transcript_path = transcript_dir / f"{ap.stem}.txt"
                    csv_output = chunks_dir / f"{ap.stem}_chunks.csv"
                    self.after(0, self._log, f"      [{i}/{n}] {ap.name}")
                    create_chunks.chunk_transcript(
                        input_path=transcript_path,
                        output_path=csv_output,
                        chunk_minutes=2,
                    )
                    chunks_map[ap] = csv_output
                    self.after(0, self._set_status,
                               f"Step 3/4 — Chunked {i}/{n}")
                
                
                #chunks_csv = tmp_path / district_name / f"{audio_path.stem}.csv"
                # from transcript_chunking import create_chunks
                # create_chunks.chunk_transcript(
                #     input_path=transcript_path,
                #     output_path=chunks_csv,
                #     chunk_minutes=2,
                # )

                # ── Step 4: Predict + build docx ──────────────────────────
                self.after(0, self._set_status, "Step 4/4 — Running model & building report…", 0.80)
                self.after(0, self._log, "[4/4] Running Logistic Regression model…")

                from research_labeling import pipeline
                # predictions = run_predictions(
                #     chunks_csv=chunks_csv,
                #     log_fn=lambda m: self.after(0, self._log, m)
                # )

                # stem = audio_path.stem
                # docx_path = out_dir / f"{stem}_highlighted.docx"

                # from pipeline.docx_builder import build_docx
                # build_docx(
                #     transcript_path=transcript_txt,
                #     predictions=predictions,
                #     output_path=docx_path,
                #     video_url=source_label
                # )

                # self._result_path = str(docx_path)
                # self.after(0, self._set_status, "Done! Transcript saved.", 1.0)
                # self.after(0, self._log, f"\n✅ Done! Saved to: {docx_path}")
                # self.after(0, self.open_btn.configure, {"state": "normal"})
                # self.after(0, self._show_preview, predictions)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # Use after() so UI update happens on the main thread
            self.after(0, self._show_error, str(e), tb)

        finally:
            self.after(0, self.run_btn.configure, {"state": "normal"})

    # ── Preview window ─────────────────────────────────────────────────────────
    def _show_preview(self, predictions: list):
        """Pop a scrollable preview window with highlighted chunks."""
        win = ctk.CTkToplevel(self)
        win.title("Transcript Preview")
        win.geometry("720x560")

        ctk.CTkLabel(
            win, text="Highlighted Transcript",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(16, 4))

        ctk.CTkLabel(
            win, text="Yellow = flagged by model   |   White = not flagged",
            font=ctk.CTkFont(size=11), text_color="gray"
        ).pack(pady=(0, 10))

        text_widget = tk.Text(
            win, wrap="word", font=("Arial", 12), padx=12, pady=8,
            relief="flat", bg="#1a1a1a", fg="white",
            selectbackground="#444"
        )
        text_widget.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        text_widget.tag_configure("highlight", background="#b5a800", foreground="black")
        text_widget.tag_configure("normal", foreground="#cccccc")
        text_widget.tag_configure("header", foreground="#888888", font=("Arial", 10))

        for chunk in predictions:
            header = f"[{chunk['window_start']} – {chunk['window_end']}]\n"
            text_widget.insert("end", header, "header")
            tag = "highlight" if chunk["flagged"] else "normal"
            text_widget.insert("end", chunk["text"].strip() + "\n\n", tag)

        text_widget.configure(state="disabled")