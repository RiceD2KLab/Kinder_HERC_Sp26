"""
colab_runner.py
Single entry point for running the full pipeline from a Colab notebook.

Two modes:
  - run(url, district)            — download from a URL, then process
  - run_from_file(path, district) — skip download, process a local audio file
"""

from pathlib import Path

from web_scraping.cli import run_scraping_pipeline
from web_scraping.models import Source
from transcription.parakeet_transcribe import run_transcription, load_asr_model
from transcript_chunking import create_chunks
from app.trained_model import run_predictions
from app.highlighter import build_docx

AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac"}


def run(
    url: str,
    district: str,
    out_dir: str = "/content/output",
    cutoff_str: str = "2024-09-01",
    max_candidates: int = 60,
    asr_model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
) -> None:
    """Run the full download → transcribe → chunk → predict → report pipeline."""
    out_dir        = Path(out_dir)
    scrape_dir     = Path("/content/scraped")
    transcript_dir = Path("/content/transcripts")
    chunks_dir     = Path("/content/chunks")

    for d in (out_dir, scrape_dir, transcript_dir, chunks_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download ───────────────────────────────────────────────────
    print("Step 1/4 — Downloading audio...")
    run_scraping_pipeline(
        frag_workers=8,
        max_candidates=max_candidates,
        include_title=".*",
        include_anchor_label=".*",
        workers=1,
        sources=[Source(district=district, url=url)],
        out_path=scrape_dir,
        cutoff_str=cutoff_str,
    )

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
    asr_model = load_asr_model(asr_model_name)

    for ap in audio_files:
        print(f"  Transcribing: {ap.name}")
        msg = run_transcription(input_path=ap, output_path=transcript_dir, asr_model=asr_model)
        if msg:
            print(f"  {msg}")

    # ── Step 3: Chunk ──────────────────────────────────────────────────────
    print("Step 3/4 — Chunking transcripts...")
    chunks_map: dict = {}

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

    # ── Step 4: Predict & report ───────────────────────────────────────────
    print("Step 4/4 — Running model and building reports...")
    for ap, chunks_csv in chunks_map.items():
        print(f"  Scoring: {ap.name}")
        predictions = run_predictions(chunks_csv=chunks_csv, log_fn=print)
        docx_path   = out_dir / f"{ap.stem}_highlighted.docx"
        build_docx(predictions=predictions, output_path=docx_path,
                   district=district, video_url=url)
        print(f"  Report saved → {docx_path}")

    print("\nAll done!")


def run_from_file(
    file_path: str,
    district: str,
    out_dir: str = "/content/output",
    asr_model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
) -> None:
    """Process a locally uploaded audio/video file instead of downloading from a URL."""
    audio_path = Path(file_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")
    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{audio_path.suffix}'. "
            f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    out_dir        = Path(out_dir)
    transcript_dir = Path("/content/transcripts")
    chunks_dir     = Path("/content/chunks")

    for d in (out_dir, transcript_dir, chunks_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Input file : {audio_path.name}")
    print(f"District   : {district}")

    # ── Step 1: Transcribe ─────────────────────────────────────────────────
    print("\nStep 1/3 — Transcribing...")
    asr_model = load_asr_model(asr_model_name)
    msg = run_transcription(input_path=audio_path, output_path=transcript_dir, asr_model=asr_model)
    if msg:
        print(f"  {msg}")

    # ── Step 2: Chunk ──────────────────────────────────────────────────────
    print("\nStep 2/3 — Chunking...")
    transcript_path = transcript_dir / f"{audio_path.stem}.txt"
    if not transcript_path.exists():
        raise RuntimeError(f"Transcription output not found: {transcript_path}")

    csv_out = chunks_dir / f"{audio_path.stem}_chunks.csv"
    result  = create_chunks.chunk_transcript(
        input_path=transcript_path,
        output_path=csv_out,
        chunk_minutes=2,
    )
    print(f"  {result}")

    # ── Step 3: Predict & report ───────────────────────────────────────────
    print("\nStep 3/3 — Running model and building report...")
    predictions = run_predictions(chunks_csv=csv_out, log_fn=print)
    docx_path   = out_dir / f"{audio_path.stem}_highlighted.docx"
    build_docx(
        predictions=predictions,
        output_path=docx_path,
        district=district,
        video_url=f"Uploaded file: {audio_path.name}",
    )
    print(f"\nDone! Report saved → {docx_path}")
