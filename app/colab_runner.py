"""
colab_runner.py
Single entry point for running the full pipeline from a Colab notebook.

Usage (in a Colab cell):
    import sys
    sys.path.insert(0, "/content/your_project_root")   # adjust to wherever you cloned
    from collab_runner import run
    run(url="https://...", district="Houston ISD")
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Make sure the project root is importable ───────────────────────────────
# __file__ is  <project_root>/collab_runner.py  →  parent is the root.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Project imports ────────────────────────────────────────────────────────
from web_scraping.cli import run_scraping_pipeline
from web_scraping.models import Source
from transcription.parakeet_transcribe import run_transcription, load_asr_model
from transcript_chunking import create_chunks          # requires transcript_chunking/__init__.py
from app.trained_model import run_predictions
from app.highlighter import build_docx

# ── Constants ──────────────────────────────────────────────────────────────
AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac"}


def run(
    url: str,
    district: str,
    out_dir: str = "/content/output",
    cutoff_str: str = "2024-09-01",
    max_candidates: int = 60,       # was 1 — that caused nearly every scrape to fail
    asr_model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
) -> None:
    """Run the full download → transcribe → chunk → predict → report pipeline.

    Parameters
    ----------
    url : str
        Meeting video URL or Swagit playlist/index page.
    district : str
        Human-readable district name (used in output filenames and the report).
    out_dir : str
        Where to write the final highlighted .docx report(s).
    cutoff_str : str
        ISO date string; meetings before this date are skipped.
    max_candidates : int
        Maximum Swagit candidate links to try when scraping a playlist page.
        Default 60 matches the pipeline's own default and is almost always correct.
    asr_model_name : str
        NeMo ASR model identifier.
    """
    out_dir      = Path(out_dir)
    scrape_dir   = Path("/content/scraped")
    transcript_dir = Path("/content/transcripts")
    chunks_dir   = Path("/content/chunks")

    for d in (out_dir, scrape_dir, transcript_dir, chunks_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download audio ─────────────────────────────────────────────
    print("Step 1/4 — Downloading audio...")
    run_scraping_pipeline(
        frag_workers=8,
        max_candidates=max_candidates,      # fixed: was hardcoded to 1
        include_title=".*",
        include_anchor_label=".*",
        workers=1,
        sources=[Source(district=district, url=url)],
        out_path=scrape_dir,
        cutoff_str=cutoff_str,
    )

    # process_source saves to  scrape_dir/<district>/  so rglob from scrape_dir covers it
    audio_files = sorted(
        f for f in scrape_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    print(f"  Found {len(audio_files)} audio file(s)")

    if not audio_files:
        print("  No audio downloaded — check the URL or cutoff date. Aborting.")
        return

    # ── Step 2: Transcribe ─────────────────────────────────────────────────
    print("Step 2/4 — Transcribing...")
    # Load the model once, outside the loop (each load takes ~30s on Colab)
    asr_model = load_asr_model(asr_model_name)

    for ap in audio_files:
        print(f"  Transcribing: {ap.name}")
        msg = run_transcription(
            input_path=ap,
            output_path=transcript_dir,
            asr_model=asr_model,
        )
        if msg:
            print(f"  {msg}")

    # ── Step 3: Chunk transcripts ──────────────────────────────────────────
    print("Step 3/4 — Chunking transcripts...")
    chunks_map: dict[Path, Path] = {}

    for ap in audio_files:
        transcript_path = transcript_dir / f"{ap.stem}.txt"
        if not transcript_path.exists():
            print(f"  WARNING: transcript not found for {ap.name}, skipping.")
            continue

        csv_out = chunks_dir / f"{ap.stem}_chunks.csv"
        result  = create_chunks.chunk_transcript(
            input_path=transcript_path,
            output_path=csv_out,
            chunk_minutes=2,
        )
        print(f"  {result}")
        chunks_map[ap] = csv_out

    if not chunks_map:
        print("  No chunks produced. Aborting.")
        return

    # ── Step 4: Predict & build reports ───────────────────────────────────
    print("Step 4/4 — Running model and building reports...")
    for ap, chunks_csv in chunks_map.items():
        print(f"  Scoring: {ap.name}")
        predictions = run_predictions(
            chunks_csv=chunks_csv,
            log_fn=print,
        )
        docx_path = out_dir / f"{ap.stem}_highlighted.docx"
        build_docx(
            predictions=predictions,
            output_path=docx_path,
            district=district,
            video_url=url,
        )
        print(f"  Report saved → {docx_path}")

    print("\nAll done!")
