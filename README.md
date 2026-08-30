# DJ Mix Alignment Pipeline

A completed audio-analysis prototype that turns a DJ mix and its track metadata into estimated cue-in and cue-out times. It combines audio recognition, feature extraction, Dynamic Time Warping (DTW), and visual reporting to make mix structure easier to inspect.

## Project status

Final prototype, prepared for portfolio review. The end-to-end workflow is implemented; it expects user-supplied or otherwise permitted tracklist and audio inputs. Raw audio, scraped pages, generated datasets, and local configuration are intentionally excluded from version control.

## The problem

Finding where an individual track enters and exits a long DJ mix is time-consuming when timestamps are missing or approximate. This project estimates those boundaries by aligning each candidate track with the full mix at beat level, then compares the inferred positions with available tracklist timestamps.

## Key features

- Parses saved tracklist HTML into structured CSV metadata.
- Samples a local mix with Shazam-based recognition to build a provisional tracklist.
- Retrieves candidate audio with `yt-dlp` when the user has permission to do so.
- Extracts MFCC and chroma features, with optional key-invariant chroma matching.
- Uses DTW paths to estimate cue points and match quality for every track.
- Produces PDF visualizations of alignment paths, cue regions, and available ground-truth markers.

## Architecture

```mermaid
flowchart LR
    A[Saved tracklist HTML or local mix audio] --> B[Scraper or recognizer]
    B --> C[Track metadata CSV]
    C --> D[Audio acquisition]
    D --> E[MP3 to WAV and feature extraction]
    E --> F[Dynamic Time Warping alignment]
    F --> G[Cue estimates and PDF visualizations]
```

The pipeline is orchestrated by `src/main.py`. `tracklist_scraper.py` and `recognizer.py` create metadata; `download.py` acquires permitted audio; `feature_extraction.py`, `align_tracks.py`, and `dtw.py` perform matching; and `visualize.py` creates the reports.

## Stack

Python, pandas, NumPy, librosa, SciPy, numba, matplotlib, pydub, Beautiful Soup, `yt-dlp`, `shazamio`, and FFmpeg.

## Quick start

Use Python 3.10+ and install FFmpeg first. On macOS, for example, run `brew install ffmpeg`.

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the full pipeline with a saved HTML tracklist that you are permitted to use:

```bash
python src/main.py --html /path/to/tracklist.html
```

Or analyze a local mix file while skipping network download:

```bash
python src/main.py --mp3 /path/to/mix.mp3 --skip-download
```

For an already prepared mix, run alignment only:

```bash
python src/main.py --mix-id <mix-id> --skip-download --skip-visualize --features mfcc --no-key-invariant
```

Use `python src/main.py --help` to see output-directory overrides and all pipeline switches.

## Testing and verification

The repository currently has no automated test suite. A release check compiles the source tree and exercises the command-line help path. Full integration runs require FFmpeg plus locally supplied, appropriately licensed audio and track metadata.

## Responsible use

Only process, download, or retain audio and tracklist data that you are authorized to use. The repository contains code only; it does not distribute music, scraped tracklists, credentials, or generated results.

## License

No license has been selected for this repository. Until a license is added, the source is not offered under an open-source license.
