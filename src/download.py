#!/usr/bin/env python3
"""
Download audio files for a mix using yt-dlp based on mix_id.

- Reads mix title from ../data/mixes.csv
- Reads tracklist from ../data/meta/[mix_id].csv
- Downloads original mix as mix.mp3 and tracks as {i_track}.mp3
- Saves to ../data/mp3/[mix_id]/
- Validates search results to ensure relevance
- Skips unidentified tracks (marked as 'ID')
"""

import argparse
import os
import sys
import pandas as pd
import yt_dlp
from fuzzywuzzy import fuzz
from urllib.parse import urlparse

def validate_mix_id(mix_id, mixes_csv):
    """Check if mix_id exists in mixes.csv and return mix title."""
    try:
        df = pd.read_csv(mixes_csv)
        if mix_id in df['mix_id'].values:
            return df[df['mix_id'] == mix_id]['mix_title'].iloc[0]
        else:
            print(f"Error: mix_id {mix_id} not found in {mixes_csv}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error reading {mixes_csv}: {e}", file=sys.stderr)
        sys.exit(1)

def validate_tracklist(mix_id, tracklist_csv):
    """Read and validate tracklist CSV for the mix_id."""
    try:
        df = pd.read_csv(tracklist_csv)
        if df.empty:
            print(f"Warning: {tracklist_csv} is empty", file=sys.stderr)
            return None
        return df
    except Exception as e:
        print(f"Error reading {tracklist_csv}: {e}", file=sys.stderr)
        return None

def is_likely_match(search_query, video_title, threshold=40):
    """
    Check if video title is likely to match the search query using fuzzy matching.
    Returns True if match score is above threshold.
    """
    score = fuzz.partial_ratio(search_query.lower(), video_title.lower())
    print('Likelihood: ', score)
    return score >= threshold

def extract_youtube_id(url):
    """Extract YouTube video ID from URL."""
    parsed = urlparse(url)
    if parsed.netloc in ('www.youtube.com', 'youtube.com', 'youtu.be'):
        if parsed.netloc == 'youtu.be':
            return parsed.path[1:]
        query = parsed.query
        for param in query.split('&'):
            if param.startswith('v='):
                return param[2:]
    return None

def download_audio(search_query, output_path, filename, mix_id):
    """
    Download audio using yt-dlp and validate relevance.
    Saves file as [output_path]/[filename].mp3.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, f'{filename}.%(ext)s'),
        'quiet': False,
        'noplaylist': True,
        'default_search': 'ytsearch',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Ensure output directory exists
            output_dir = os.path.join('../data/mp3', str(mix_id))
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{filename}.mp3")
            
            if os.path.exists(output_file):
                print(f"Info: File already exists at {output_file}. Skipping download.", file=sys.stderr)
                return True
            
            # Get info without downloading
            info = ydl.extract_info(f"ytsearch:{search_query}", download=False)
            if not info or 'entries' not in info or not info['entries']:
                print(f"Warning: No results found for '{search_query}'", file=sys.stderr)
                return False

            video = info['entries'][0]
            video_title = video.get('title', '')
            video_url = video.get('webpage_url', '')

            # Validate relevance
            if not is_likely_match(search_query, video_title):
                print(f"Warning: Top result '{video_title}' for '{search_query}' seems unrelated (match score too low). Skipping.", file=sys.stderr)
                return False

            # Download
            print(f"Downloading '{video_title}' for '{search_query}' as {filename}.mp3")
            ydl.download([video_url])
            
            return True

    except Exception as e:
        print(f"Error downloading '{search_query}': {e}", file=sys.stderr)
        return False

def run_downloader(mix_id, mixes_csv_path='../data/mixes.csv', meta_dir='../data/meta', mp3_base_dir='../data/mp3'):
    """
    Runs the full download process for a given mix ID.

    Args:
        mix_id (str): The ID of the mix to download audio for.
        mixes_csv_path (str, optional): Path to the mixes database CSV. Defaults to '../data/mixes.csv'.
        meta_dir (str, optional): Directory containing metadata CSVs. Defaults to '../data/meta'.
        mp3_base_dir (str, optional): Base directory to save MP3 files. Defaults to '../data/mp3'.

    Returns:
        str: The path to the directory containing downloaded MP3s (e.g., '../data/mp3/[mix_id]')
             or None if a critical error occurs (e.g., mix_id not found).
    """
    mix_id = str(mix_id)  # Ensure mix_id is a string
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct full paths relative to the script location
    full_mixes_csv_path = os.path.join(script_dir, mixes_csv_path)
    full_tracklist_csv_path = os.path.join(script_dir, meta_dir, f"{mix_id}.csv")
    full_output_dir = os.path.join(script_dir, mp3_base_dir, mix_id)

    # Validate mix_id and get mix title
    try:
        mix_title = validate_mix_id(mix_id, full_mixes_csv_path)
    except SystemExit: # Catch SystemExit from validate_mix_id on error
        return None

    # Download original mix as mix.mp3
    print(f"--- Downloading Mix: {mix_title} ---")
    mix_downloaded = download_audio(mix_title, full_output_dir, "mix", mix_id)
    if not mix_downloaded:
        print(f"Warning: Failed to download original mix '{mix_title}'", file=sys.stderr)
        # Continue to download tracks even if mix download fails, but maybe return None later?
        # For now, let's proceed but the user should be aware.
    else:
        print(f"Successfully processed original mix (downloaded or existed).")

    # Validate and download tracks
    print(f"--- Downloading Tracks for Mix ID: {mix_id} ---")
    tracklist_df = validate_tracklist(mix_id, full_tracklist_csv_path)
    if tracklist_df is None:
        print(f"Warning: Could not read or validate tracklist at {full_tracklist_csv_path}. Skipping track downloads.", file=sys.stderr)
    else:
        tracks_downloaded_count = 0
        tracks_failed_count = 0
        for _, track in tracklist_df.iterrows():
            artist = str(track['artist'])  # Ensure string
            title = str(track['title'])   # Ensure string
            try:
                i_track = int(track['i_track'])  # Convert to int safely
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid i_track value for track {artist} - {title}: {e}. Skipping.", file=sys.stderr)
                tracks_failed_count += 1
                continue

            # Skip unidentified tracks
            if 'ID' == title.upper() or 'ID' == artist.upper():
                print(f"Info: Skipping unidentified track: {artist} - {title}", file=sys.stderr)
                continue

            search_query = f"{artist} {title}"
            print(f"Processing track {i_track}: {search_query}")
            if not download_audio(search_query, full_output_dir, str(i_track), mix_id):
                print(f"Warning: Failed to download track '{search_query}'", file=sys.stderr)
                tracks_failed_count += 1
            else:
                print(f"Successfully processed track {i_track}.")
                tracks_downloaded_count += 1
        print(f"--- Track Download Summary: {tracks_downloaded_count} processed, {tracks_failed_count} failed/skipped ---")

    # Return the output directory path even if some downloads failed,
    # as subsequent steps might still work with partial data.
    return full_output_dir

# Example usage (if run directly, though intended as module):
# if __name__ == "__main__":
#     test_mix_id = '6qdzkf9' # Replace with a valid mix ID
#     download_dir = run_downloader(test_mix_id)
#     if download_dir:
#         print(f"Download process finished. Files are in: {download_dir}")
#     else:
#         print(f"Download process failed for mix ID: {test_mix_id}")
