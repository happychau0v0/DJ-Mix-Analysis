# DJ Mix Analysis Pipeline

## Overview

This project provides a pipeline to analyze DJ mixes sourced from 1001tracklists.com. It automates the process of:

1.  **Scraping:** Extracting tracklist metadata (artist, title, timestamp) from a saved 1001tracklists HTML page.
2.  **Downloading:** Fetching the audio for the full mix and individual tracks using YouTube search (`yt-dlp`).
3.  **Aligning:** Aligning each individual track's audio to the full mix audio using Dynamic Time Warping (DTW) on audio features (MFCC, Chroma) to find precise cue-in and cue-out points.
4.  **Visualizing:** Generating a PDF plot showing the alignment paths, calculated cue points, and ground truth timestamps (if available).

## Features

*   Parses 1001tracklists HTML for track metadata.
*   Automated audio downloading via YouTube search.
*   Robust track-to-mix alignment using DTW.
*   Key-invariant alignment option for handling pitch shifts.
*   Parallel processing for faster alignment.
*   Feature caching (`joblib`) to speed up repeated analysis.
*   Clear visualization of alignment results.
*   Streamlined pipeline execution via `src/main.py`.
*   Command-line interface with options to control features, key invariance, and skip steps.

## Workflow

The analysis follows these steps, orchestrated by `src/main.py`:

1.  **Scrape (`src/tracklist_scraper.py`):**
    *   Input: HTML file from `tracklists/`.
    *   Output: Mix ID, metadata CSV (`data/meta/[mix_id].csv`), updated `data/mixes.csv`.
2.  **Download (`src/download.py`):**
    *   Input: Mix ID, metadata CSV.
    *   Output: Mix audio (`data/mp3/[mix_id]/mix.mp3`), track audio (`data/mp3/[mix_id]/[i_track].mp3`).
3.  **Align (`src/align_tracks.py`):**
    *   Input: Mix ID, MP3 audio files.
    *   Process: Converts MP3 to WAV (`data/wav/`), extracts features, performs DTW.
    *   Output: Alignment results PKL file (`data/align/[mix_id]-[features]-[keyinv].pkl`).
4.  **Visualize (`src/visualize.py`):**
    *   Input: Mix ID, alignment PKL file, metadata CSV.
    *   Output: Visualization PDF (`data/dtwviz/[mix_id].pdf`).

## Directory Structure

```
.
├── data/
│   ├── align/      # Stores alignment results (.pkl)
│   ├── dtwviz/     # Stores visualization PDFs
│   ├── meta/       # Stores tracklist metadata (.csv)
│   ├── mixes.csv   # Database of processed mixes
│   ├── mp3/        # Stores downloaded audio (.mp3)
│   └── wav/        # Stores converted audio (.wav)
├── src/
│   ├── cache/      # Joblib cache for features
│   ├── align_tracks.py
│   ├── download.py
│   ├── main.py     # Main pipeline script
│   ├── tracklist_scraper.py
│   └── visualize.py
├── tracklists/     # Input HTML files from 1001tracklists.com
├── README.md       # This file
└── requirement.txt # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirement.txt
    ```
3.  **Install FFmpeg:** `yt-dlp` requires FFmpeg for audio conversion. Follow instructions for your OS (e.g., `brew install ffmpeg` on macOS, `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu).
4.  **Add HTML files:** Place saved 1001tracklists.com HTML pages into the `tracklists/` directory.

## Usage

The main pipeline is executed using `src/main.py`.

```bash
python src/main.py [options]
```

**Required Arguments (choose one):**

*   `--html <filename>`: Start the pipeline by scraping the specified HTML file (e.g., `--html avicii` for `tracklists/avicii.html`).
*   `--mix-id <mix_id>`: Start the pipeline using an existing Mix ID, skipping the scraping step (e.g., `--mix-id 6qdzkf9`).

**Optional Arguments:**

*   `--features <list>`: Comma-separated list of features for alignment (e.g., `chroma`, `mfcc`, `chroma,mfcc`). Default: `chroma,mfcc`.
*   `--key-invariant`: Use key-invariant matching during alignment (default).
*   `--no-key-invariant`: Disable key-invariant matching.
*   `--skip-download`: Skip the audio download step.
*   `--skip-align`: Skip the track alignment step.
*   `--skip-visualize`: Skip the visualization step.
*   `--meta-dir <path>`: Override default metadata directory (`../data/meta`).
*   `--mp3-dir <path>`: Override default MP3 directory (`../data/mp3`).
*   `--wav-dir <path>`: Override default WAV directory (`../data/wav`).
*   `--align-dir <path>`: Override default alignment results directory (`../data/align`).
*   `--viz-dir <path>`: Override default visualization directory (`../data/dtwviz`).
*   `--mixes-db <path>`: Override default mixes database path (`../data/mixes.csv`).

## Example Usage

*   **Run the full pipeline for a new mix:**
    ```bash
    python src/main.py --html avicii
    ```
*   **Run alignment and visualization for an existing mix ID:**
    ```bash
    python src/main.py --mix-id 6qdzkf9 --skip-download
    ```
*   **Run alignment only, using only MFCC features and no key invariance:**
    ```bash
    python src/main.py --mix-id 6qdzkf9 --skip-download --skip-visualize --features mfcc --no-key-invariant
    ```

## Output Files

The pipeline generates files in the `data/` subdirectories:

*   `data/meta/[mix_id].csv`: Tracklist metadata.
*   `data/mp3/[mix_id]/`: Downloaded MP3 audio files.
*   `data/wav/[mix_id]/`: Converted WAV audio files (used for alignment).
*   `data/align/[mix_id]-[features]-[keyinv].pkl`: Alignment results (Python pickle format).
*   `data/dtwviz/[mix_id].pdf`: Visualization of the alignment.
*   `data/mixes.csv`: Updated list of processed mixes.

## Dependencies

See `requirement.txt` for a full list of Python packages. Key dependencies include:

*   `pandas`
*   `beautifulsoup4`
*   `yt-dlp`
*   `fuzzywuzzy`
*   `librosa`
*   `matplotlib`
*   `joblib`
*   `pydub`
*   `numpy`

External dependency:

*   `ffmpeg`
