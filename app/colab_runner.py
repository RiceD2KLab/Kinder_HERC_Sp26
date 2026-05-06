"""
colab_runner.py
Single entry point for running the full pipeline from a Colab notebook.
Supports two modes:
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


def run(url: str, district: str, out_dir: str = "/content/output"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scrape_dir = Path("/content/scraped")
    scrape_dir.mkdir(exist_ok=True)
    transcript_dir = Path("/content/transcripts")
    transcript_dir.mkdir(exist_ok=True)
    chunks_dir = Path("/content/chunks")
    chunks_dir.mkdir(exist_ok=True)

    print("Step 1/4 — Downloading audio...")
    run_scraping_pipeline(
        frag_workers=8, max_candidates=1,
        include_title=".*", include_anchor_label=".*",
        workers=1,
        sources=[Source(district=district, url=url)],
        out_path=scrape_dir,
        cutoff_str="2024-09-01"
    )
    audio_files = sorted(f for f in scrape_dir.rglob("*")
                         if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS)
    print(f"Found {len(audio_files)} audio file(s)")

    print("Step 2/4 — Transcribing...")
    asr_model = load_asr_model("nvidia/parakeet-tdt-0.6b-v3")
    for ap in audio_files:
        run_transcription(input_path=ap, output_path=transcript_dir, asr_model=asr_model)

    print("Step 3/4 — Chunking...")
    chunks_map = {}
    for ap in audio_files:
        csv_out = chunks_dir / f"{ap.stem}_chunks.csv"
        create_chunks.chunk_transcript(
            input_path=transcript_dir / f"{ap.stem}.txt",
            output_path=csv_out,
            chunk_minutes=2
        )
        chunks_map[ap] = csv_out

    print("Step 4/4 — Running model and building report...")
    for ap in audio_files:
        predictions = run_predictions(chunks_csv=chunks_map[ap])
        docx_path = out_dir / f"{ap.stem}_highlighted.docx"
        build_docx(predictions=predictions, output_path=docx_path,
                   district=district, video_url=url)
        print(f"Done! Report saved to: {docx_path}")


def run_from_file(file_path: str, district: str, out_dir: str = "/content/output"):
    """Process a locally uploaded audio/video file instead of downloading from a URL.

    Parameters
    ----------
    file_path : str
        Path to the uploaded audio/video file (wav, mp3, mp4, m4a, flac, ogg, aac).
    district : str
        District name used to label the output report.
    out_dir : str
        Directory where the highlighted .docx report will be saved.
    """
    audio_path = Path(file_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")
    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{audio_path.suffix}'. "
            f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = Path("/content/transcripts")
    transcript_dir.mkdir(exist_ok=True)
    chunks_dir = Path("/content/chunks")
    chunks_dir.mkdir(exist_ok=True)

    print(f"Input file : {audio_path.name}")
    print(f"District   : {district}")

    print("\nStep 1/3 — Transcribing...")
    asr_model = load_asr_model("nvidia/parakeet-tdt-0.6b-v3")
    run_transcription(input_path=audio_path, output_path=transcript_dir, asr_model=asr_model)

    print("\nStep 2/3 — Chunking...")
    csv_out = chunks_dir / f"{audio_path.stem}_chunks.csv"
    create_chunks.chunk_transcript(
        input_path=transcript_dir / f"{audio_path.stem}.txt",
        output_path=csv_out,
        chunk_minutes=2
    )

    print("\nStep 3/3 — Running model and building report...")
    predictions = run_predictions(chunks_csv=csv_out)
    docx_path = out_dir / f"{audio_path.stem}_highlighted.docx"
    build_docx(
        predictions=predictions,
        output_path=docx_path,
        district=district,
        video_url=f"Uploaded file: {audio_path.name}",
    )
    print(f"\nDone! Report saved to: {docx_path}")
