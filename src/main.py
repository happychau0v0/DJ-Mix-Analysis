import argparse
import os
import sys
import tracklist_scraper
import download
import align_tracks
import visualize
import recognizer
import asyncio

def main():
    parser = argparse.ArgumentParser(description='Run the DJ Mix Analysis pipeline.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--html', help='HTML file base name in ../tracklists/ (e.g., "avicii").')
    input_group.add_argument('--mix-id', help='Mix ID to process (skips scraping).')
    input_group.add_argument('--mp3', help='Path to MP3 file for mix analysis.')
    parser.add_argument('--features', default='chroma,mfcc', help='Alignment features (comma-separated).')
    key_inv_group = parser.add_mutually_exclusive_group()
    key_inv_group.add_argument('--key-invariant', action='store_true', default=True, help='Use key-invariant matching.')
    key_inv_group.add_argument('--no-key-invariant', dest='key_invariant', action='store_false', help='Disable key-invariant matching.')
    parser.add_argument('--skip-download', action='store_true', help='Skip audio download.')
    parser.add_argument('--skip-align', action='store_true', help='Skip track alignment.')
    parser.add_argument('--skip-visualize', action='store_true', help='Skip visualization.')
    parser.add_argument('--meta-dir', default='../data/meta', help='Metadata CSV directory.')
    parser.add_argument('--mp3-dir', default='../data/mp3', help='MP3 download directory.')
    parser.add_argument('--wav-dir', default='../data/wav', help='WAV conversion directory.')
    parser.add_argument('--align-dir', default='../data/align', help='Alignment PKL directory.')
    parser.add_argument('--viz-dir', default='../data/dtwviz', help='Visualization PDF directory.')
    parser.add_argument('--mixes-db', default='../data/mixes.csv', help='Mixes database CSV path.')
    args = parser.parse_args()

    current_mix_id = args.mix_id
    meta_csv_path = None
    pkl_path = None
    pdf_path = None
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.html:
        print(f"\n--- Step 1: Scraping ---")
        if not os.path.exists(args.html):
            print(f"Tracklist HTML file not found at {args.html}. Exiting.", file=sys.stderr)
            sys.exit(1)
        current_mix_id, meta_csv_path = tracklist_scraper.run_scraper(args.html, mixes_db_path=args.mixes_db)
        if not current_mix_id:
            print("Scraping failed. Exiting.", file=sys.stderr)
            sys.exit(1)
        print(f"Scraping successful. Mix ID: {current_mix_id}")
    elif args.mp3:
        print(f"\n--- Step 1: Song Detection ---")
        if not os.path.exists(args.mp3):
            print(f"MP3 file not found at {args.mp3}. Exiting.", file=sys.stderr)
            sys.exit(1)
        current_mix_id, meta_csv_path = asyncio.run(recognizer.detect_songs_in_mix(args.mp3, 60, 15))
        if not current_mix_id:
            print("Song detection failed. Exiting.", file=sys.stderr)
            sys.exit(1)
        print(f"Song detection successful. Mix ID: {current_mix_id}")
    else:
        print(f"\n--- Step 1: Scraping (Skipped) ---")
        print(f"Using Mix ID: {current_mix_id}")
        meta_csv_path = os.path.join(script_dir, args.meta_dir, f"{current_mix_id}.csv")
        if not os.path.exists(meta_csv_path):
            print(f"Warning: Metadata CSV not found at {meta_csv_path}.", file=sys.stderr)

    if not args.skip_download:
        print(f"\n--- Step 2: Downloading ---")
        download_dir = download.run_downloader(current_mix_id, mixes_csv_path=args.mixes_db, meta_dir=args.meta_dir, mp3_base_dir=args.mp3_dir)
        if not download_dir:
            print("Download failed. Exiting.", file=sys.stderr)
            sys.exit(1)
        print(f"Download finished. Files in: {download_dir}")
    else:
        print(f"\n--- Step 2: Downloading (Skipped) ---")

    if not args.skip_align:
        print(f"\n--- Step 3: Alignment ---")
        features_list = args.features.split(',')
        print(f"Aligning tracks. Features: {features_list}, Key Invariant: {args.key_invariant}")
        pkl_path = align_tracks.process_mix(current_mix_id, features=features_list, key_invariant=args.key_invariant)
        if not pkl_path or not os.path.exists(pkl_path):
            print("Alignment failed. Exiting.", file=sys.stderr)
            sys.exit(1)
        print(f"Alignment successful. Results: {pkl_path}")
    else:
        print(f"\n--- Step 3: Alignment (Skipped) ---")
        if not args.skip_visualize:
            feature_id = '+'.join(args.features.split(','))
            pkl_filename = f"{current_mix_id}-{feature_id}{'-keyinv' if args.key_invariant else ''}.pkl"
            pkl_path = os.path.join(script_dir, args.align_dir, pkl_filename)
            print(f"Assuming alignment results at: {pkl_path}")

    if not args.skip_visualize:
        print(f"\n--- Step 4: Visualization ---")
        if not pkl_path:
            print("No alignment results for visualization. Exiting.", file=sys.stderr)
            sys.exit(1)
        print(f"Generating visualization using {pkl_path}")
        pdf_path = visualize.run_visualizer(current_mix_id, pkl_path=pkl_path, viz_base_dir=args.viz_dir)
        if not pdf_path:
            print("Visualization failed.", file=sys.stderr)
        else:
            print(f"Visualization successful. PDF: {pdf_path}")
    else:
        print(f"\n--- Step 4: Visualization (Skipped) ---")

    print(f"\n--- Pipeline Finished for Mix ID: {current_mix_id} ---")
    if meta_csv_path: print(f"Metadata CSV: {meta_csv_path}")
    if pkl_path: print(f"Alignment PKL: {pkl_path}")
    if pdf_path: print(f"Visualization PDF: {pdf_path}")

if __name__ == "__main__":
    main()