# DJ Mix Analysis Pipeline

## Overview

This project provides a pipeline to analyze DJ mixes sourced from 1001tracklists.com. It automates the process of:

1.  **Scraping / Recognizing:** Extracting tracklist metadata from a saved 1001tracklists HTML page, or recognize songs from a provided MP3.
2.  **Downloading:** Fetching the audio for the full mix and individual tracks using YouTube search (`yt-dlp`).
3.  **Aligning:** Aligning each individual track's audio to the full mix audio using Dynamic Time Warping (DTW) on audio features (MFCC, Chroma) to find precise cue-in and cue-out points.
4.  **Visualizing:** Generating a PDF plot showing the alignment paths, calculated cue points, and ground truth timestamps (if available).

## Workflow

The analysis follows these steps, managed by `src/main.py`:

1.  **Scrape / Recognize:**
    *   Input: HTML file from 1001tracklists.com or MP3 file from a DJ mix.
    *   Output: Mix ID, metadata CSV (`data/meta/[mix_id].csv`), updated `data/mixes.csv`.
2.  **Download:**
    *   Input: Mix ID, metadata CSV.
    *   Output: Mix audio (`data/mp3/[mix_id]/mix.mp3`), track audio (`data/mp3/[mix_id]/[i_track].mp3`).
3.  **Align:**
    *   Input: Mix ID, MP3 audio files.
    *   Process: Converts MP3 to WAV (`data/wav/`), extracts features, performs DTW.
    *   Output: Alignment results PKL file (`data/align/[mix_id]-[features]-[keyinv].pkl`).
4.  **Visualize:**
    *   Input: Mix ID, alignment PKL file, metadata CSV.
    *   Output: Visualization PDF (`data/dtwviz/[mix_id].pdf`).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install FFmpeg:** `yt-dlp` requires FFmpeg for audio conversion. `brew install ffmpeg` on macOS.

3.  **Install Rust:** The song detection feature uses the `shazamio` library, which requires Rust. Install Rust from https://www.rust-lang.org/tools/install.

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main pipeline is executed using `src/main.py`.

```bash
python src/main.py [options]
```

### Command-Line Options

**Required Arguments (choose one):**:
- `--html HTML`: Path to the HTML file containing the tracklist (e.g., `../tracklists/avicii.html`).
- `--mp3 MP3`: Path to an MP3 file for song detection.
- `--mix-id MIX_ID`: Mix ID to process, skipping the scraping step.

**Analysis options**:
- `--features FEATURES`: Comma-separated list of alignment features (default: `chroma,mfcc`).
- `--key-invariant`: Use key-invariant matching (default: enabled).
- `--no-key-invariant`: Disable key-invariant matching.

**Pipeline options**:
- `--skip-download`: Skip the audio download step.
- `--skip-align`: Skip the track alignment step.
- `--skip-visualize`: Skip the visualization step.

**Directory overriding**:
- `--meta-dir META_DIR`: Directory for metadata CSV files (default: `../data/meta`).
- `--mp3-dir MP3_DIR`: Directory for MP3 downloads (default: `../data/mp3`).
- `--wav-dir WAV_DIR`: Directory for WAV conversions (default: `../data/wav`).
- `--align-dir ALIGN_DIR`: Directory for alignment PKL files (default: `../data/align`).
- `--viz-dir VIZ_DIR`: Directory for visualization PDFs (default: `../data/dtwviz`).
- `--mixes-db MIXES_DB`: Path to the mixes database CSV (default: `../data/mixes.csv`).

## Example Usage

*   **Run the full pipeline for a given tracklist:**
    ```bash
    python src/main.py --html avicii
    ```
*   **Detect songs & analysis for an MP3 mix:**
    ```bash
    bashpython main.py --mp3 path/to/mix.mp3
    ```
*   **Run alignment and visualization for an existing mix ID:**
    ```bash
    python src/main.py --mix-id 6qdzkf9 --skip-download
    ```
*   **Run alignment only, using only MFCC features and no key invariance:**
    ```bash
    python src/main.py --mix-id 6qdzkf9 --skip-download --skip-visualize --features mfcc --no-key-invariant
    ```

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
│   ├── database.py
│   ├── download.py
│   ├── dtw.py
│   ├── feature_extraction.py
│   ├── main.py     # Main pipeline script
│   ├── recognizer.py
│   ├── tracklist_scraper.py
│   └── visualize.py
├── README.md       # This file
└── requirements.txt # Python dependencies
```