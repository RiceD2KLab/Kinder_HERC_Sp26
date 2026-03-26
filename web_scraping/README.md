# Web Scraping - School Board Meeting Downloader

Downloads school board meeting videos and extracts audio as WAV files. Supports direct Swagit video URLs as well as any district webpage that embeds Swagit video links — including paginated archives, Bootstrap tab panes, and iframe-embedded players.

## Overview

You provide a district name and a URL. The pipeline figures out what kind of URL it is and handles the rest:

- **Direct Swagit URL** — downloaded immediately.
- **District webpage** (e.g. `katyisd.org`, `houstonisd.org`, `springbranchisd.com`) — the page is scraped for embedded Swagit links, which are then downloaded one by one.

Output files are named:

```
School Board Meetings/<District>/DISTRICT-YYYY-MM-DD-Title.wav
```

## Setup

```bash
cd web_scraping
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

**Also required:** `ffmpeg` must be installed and available on PATH.

**Important:** Always run the CLI from the `Kinder_HERC_Sp26` parent directory, not from inside `web_scraping/`:

```bash
cd Kinder_HERC_Sp26
python -m web_scraping.cli --source "Katy ISD|https://..." --cutoff 2024-09-01
```

## Usage

Each source is specified as `"District Name|URL"`. The URL can be:

- A direct Swagit video link: `https://springbranchisdtx.new.swagit.com/videos/364571/download`
- A district board meetings page: `https://www.katyisd.org/board/board/board-meeting-videos`

### Inline sources

```bash
python -m web_scraping.cli \
    --source "Katy ISD|https://www.katyisd.org/board/board/board-meeting-videos" \
    --cutoff 2024-09-01
```

Multiple sources:

```bash
python -m web_scraping.cli \
    --source "Spring Branch ISD|https://www.springbranchisd.com/about/board-of-trustees/meetings/board-meeting-videos" \
    --source "Houston ISD|https://www.houstonisd.org/board-governance/board-meetings" \
    --cutoff 2024-09-01
```

### From a sources file

Create a text file with one `District|URL` per line (blank lines and `#` comments are ignored):

```text
# sources.txt
Katy ISD|https://www.katyisd.org/board/board/board-meeting-videos
Spring Branch ISD|https://www.springbranchisd.com/about/board-of-trustees/meetings/board-meeting-videos
Houston ISD|https://www.houstonisd.org/board-governance/board-meetings
```

Run:

```bash
python -m web_scraping.cli --sources-file sources.txt --cutoff 2024-09-01
```

### Key Options

| Flag | Description | Default |
|------|-------------|---------|
| `--cutoff` | Only download meetings on or after this date (YYYY-MM-DD) | `2024-09-01` |
| `--out` | Output root directory | `School Board Meetings` |
| `--workers` | Concurrent source threads | `min(8, CPU count)` |
| `--max-candidates` | Max video links to attempt per page | `60` |
| `--frag-workers` | Concurrent fragment downloads for streams | `8` |
| `--include-title` | Regex: only download videos whose title matches | Board meeting variants |
| `--ffmpeg-location` | Path to ffmpeg binary or bin directory | Auto-detect from PATH |

Run `python -m web_scraping.cli --help` for all options.

## Output

```
School Board Meetings/
├── Katy ISD/
│   ├── Katy_ISD-2024-09-23-Board_Meetings.wav
│   └── Katy_ISD-2024-12-09-Board_Meetings.wav
├── Spring Branch ISD/
│   ├── Spring_Branch_ISD-2024-09-23-Board_Meetings.wav
│   └── Spring_Branch_ISD-2024-12-13-Special_Board_Meetings.wav
└── Houston ISD/
    └── Houston_ISD-2024-10-10-Board_Meetings.wav
```

Each WAV file is mono 16-bit PCM audio extracted from the original video.

## How It Works

1. **Direct Swagit URL** — If the source URL is already a Swagit video link, it is downloaded immediately via `yt-dlp`.
2. **District webpage** — If the URL is a district website, the page HTML is fetched and scanned for embedded Swagit video links. The scraper handles:
   - Anchor links and iframe embeds
   - Swagit-hosted iframes with paginated archives
   - Bootstrap tab panes with multiple meeting tables
3. **Download + extract** — Each candidate URL is passed to `yt-dlp`, which downloads the video and extracts WAV audio via its ffmpeg postprocessor.
4. **Rename** — Output files are renamed to `DISTRICT-YYYY-MM-DD-Title.wav` using dates parsed from video metadata, table rows, or titles.

## Module Structure

| File | Purpose |
|------|---------|
| `cli.py` | Command-line interface and argument parsing |
| `config.py` | HTTP headers and date-parsing regex patterns |
| `models.py` | `Source` dataclass (district + URL pair) |
| `dates.py` | Date extraction from video titles and URLs |
| `html_scrape.py` | HTML fetching, Swagit link extraction, pagination, tab pane handling |
| `pipeline.py` | Per-source orchestration (direct URL vs. page scraping) |
| `ytdlp_runner.py` | yt-dlp download, ffmpeg audio extraction, file renaming |
