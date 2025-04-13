#!/usr/bin/env python3
"""
Main pipeline script for DJ Mix Analysis.

Orchestrates the workflow:
1. Scrape tracklist from HTML -> Get mix_id and metadata CSV.
2. Download mix and track audio -> Get MP3 files.
3. Align tracks to mix using DTW -> Get alignment PKL file.
4. Visualize alignment -> Get PDF visualization.
"""

import argparse
import os
import sys

# Import refactored functions from other modules
try:
    import tracklist_scraper
    import download
    import align_tracks
    import visualize
except ImportError as e:
    print(f"Error importing modules: {e}", file=sys.stderr)
    print("Please ensure tracklist_scraper.py, download.py, align_tracks.py, and visualize.py are in the same directory.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run the DJ Mix Analysis pipeline.')

    # Input source: either HTML file or existing mix_id
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--html', help='Base name of the HTML file in ../tracklists/ (e.g., "avicii") to start scraping.')
    input_group.add_argument('--mix-id', help='Existing Mix ID to start processing from (skips scraping).')

    # Alignment options
    parser.add_argument('--features', default='chroma,mfcc', help='Feature types for alignment (comma-separated, e.g., "chroma,mfcc"). Default: chroma,mfcc')
    key_inv_group = parser.add_mutually_exclusive_group()
    key_inv_group.add_argument('--key-invariant', action='store_true', default=True, help='Use key-invariant matching for alignment (default).')
    key_inv_group.add_argument('--no-key-invariant', dest='key_invariant', action='store_false', help='Do not use key-invariant matching.')

    # Skip flags
    parser.add_argument('--skip-download', action='store_true', help='Skip the audio download step.')
    parser.add_argument('--skip-align', action='store_true', help='Skip the track alignment step.')
    parser.add_argument('--skip-visualize', action='store_true', help='Skip the visualization step.')

    # Path overrides (optional)
    parser.add_argument('--meta-dir', default='../data/meta', help='Directory for metadata CSVs.')
    parser.add_argument('--mp3-dir', default='../data/mp3', help='Base directory for MP3 downloads.')
    parser.add_argument('--wav-dir', default='../data/wav', help='Base directory for WAV conversions.')
    parser.add_argument('--align-dir', default='../data/align', help='Directory for alignment PKL files.')
    parser.add_argument('--viz-dir', default='../data/dtwviz', help='Directory for visualization PDFs.')
    parser.add_argument('--mixes-db', default='../data/mixes.csv', help='Path to the mixes database CSV.')


    args = parser.parse_args()

    current_mix_id = args.mix_id
    meta_csv_path = None
    pkl_path = None
    pdf_path = None

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 1. Scraping ---
    if args.html:
        print(f"\n--- Step 1: Scraping ---")
        print(f"Scraping HTML file: {args.html}.html")
        current_mix_id, meta_csv_path = tracklist_scraper.run_scraper(
            args.html,
            mixes_db_path=args.mixes_db
        )
        if not current_mix_id:
            print("Scraping failed. Exiting pipeline.", file=sys.stderr)
            sys.exit(1)
        print(f"Scraping successful. Mix ID: {current_mix_id}")
    elif args.mix_id:
        print(f"\n--- Step 1: Scraping (Skipped) ---")
        print(f"Using provided Mix ID: {current_mix_id}")
        # Construct expected meta path if scraping is skipped
        meta_csv_path = os.path.join(script_dir, args.meta_dir, f"{current_mix_id}.csv")
        if not os.path.exists(meta_csv_path):
             print(f"Warning: Metadata CSV not found at {meta_csv_path}. Subsequent steps might fail.", file=sys.stderr)
    else:
        # This case should not happen due to mutually_exclusive_group(required=True)
        print("Error: No input source specified (--html or --mix-id).", file=sys.stderr)
        sys.exit(1)


    # --- 2. Downloading ---
    if not args.skip_download:
        print(f"\n--- Step 2: Downloading ---")
        print(f"Downloading audio for Mix ID: {current_mix_id}")
        download_dir = download.run_downloader(
            current_mix_id,
            mixes_csv_path=args.mixes_db,
            meta_dir=args.meta_dir,
            mp3_base_dir=args.mp3_dir
        )
        if not download_dir:
            print("Download step failed or encountered critical errors. Exiting pipeline.", file=sys.stderr)
            # Decide if pipeline should stop. For now, let's exit.
            sys.exit(1)
        print(f"Download process finished. Files should be in: {download_dir}")
    else:
        print(f"\n--- Step 2: Downloading (Skipped) ---")


    # --- 3. Alignment ---
    if not args.skip_align:
        print(f"\n--- Step 3: Alignment ---")
        features_list = args.features.split(',')
        print(f"Aligning tracks for Mix ID: {current_mix_id}")
        print(f"Using features: {features_list}, Key Invariant: {args.key_invariant}")
        # Note: align_tracks.process_mix handles its own path construction based on mix_id, features, key_inv
        # It uses relative paths like ../data/mp3, ../data/wav, ../data/meta, ../data/align internally
        # We might need to adjust align_tracks if we want full control via args here, but let's use its defaults for now.
        # The function expects directories relative to its own location.
        pkl_path = align_tracks.process_mix(
            current_mix_id,
            features=features_list,
            key_invariant=args.key_invariant
            # We could pass mp3_dir, wav_dir, meta_path, results_dir if we modify process_mix
        )
        if not pkl_path or not os.path.exists(pkl_path):
            print("Alignment step failed or did not produce a result file. Exiting pipeline.", file=sys.stderr)
            sys.exit(1)
        print(f"Alignment successful. Results saved to: {pkl_path}")
    else:
        print(f"\n--- Step 3: Alignment (Skipped) ---")
        # If alignment is skipped, try to construct the expected pkl_path for visualization
        if not args.skip_visualize:
             feature_id = '+'.join(args.features.split(','))
             pkl_filename = f"{current_mix_id}-{feature_id}"
             if args.key_invariant:
                 pkl_filename += '-keyinv'
             pkl_filename += '.pkl'
             pkl_path = os.path.join(script_dir, args.align_dir, pkl_filename)
             print(f"Assuming alignment results are at: {pkl_path}")


    # --- 4. Visualization ---
    if not args.skip_visualize:
        print(f"\n--- Step 4: Visualization ---")
        if not pkl_path:
            print("Cannot visualize without alignment results (.pkl file path). Please run alignment or provide the path.", file=sys.stderr)
            sys.exit(1)
        print(f"Generating visualization for Mix ID: {current_mix_id} using {pkl_path}")
        pdf_path = visualize.run_visualizer(
            current_mix_id,
            pkl_path=pkl_path,
            viz_base_dir=args.viz_dir
        )
        if not pdf_path:
            print("Visualization step failed.", file=sys.stderr)
            # Don't necessarily exit, maybe just warn
        else:
            print(f"Visualization successful. PDF saved to: {pdf_path}")
    else:
        print(f"\n--- Step 4: Visualization (Skipped) ---")


    print(f"\n--- Pipeline Finished for Mix ID: {current_mix_id} ---")
    if meta_csv_path: print(f"Metadata CSV: {meta_csv_path}")
    if pkl_path: print(f"Alignment PKL: {pkl_path}")
    if pdf_path: print(f"Visualization PDF: {pdf_path}")

if __name__ == "__main__":
    main()
