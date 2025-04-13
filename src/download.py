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

def main():
    parser = argparse.ArgumentParser(description='Download audio for a mix using yt-dlp based on mix_id')
    parser.add_argument('mix_id', help='Mix ID to download audio for')
    args = parser.parse_args()

    mix_id = str(args.mix_id)  # Ensure mix_id is a string
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    mixes_csv = os.path.join(script_dir, '..', 'data', 'mixes.csv')
    tracklist_csv = os.path.join(script_dir, '..', 'data', 'meta', f"{mix_id}.csv")
    output_dir = os.path.join(script_dir, '..', 'data', 'mp3', mix_id)

    # Validate mix_id and get mix title
    mix_title = validate_mix_id(mix_id, mixes_csv)
    
    # Download original mix as mix.mp3
    print(f"Searching for original mix: {mix_title}")
    if not download_audio(mix_title, output_dir, "mix", mix_id):
        print(f"Warning: Failed to download original mix '{mix_title}'", file=sys.stderr)
    else:
        print(f"Successfully downloaded original mix as ../data/mp3/{mix_id}/mix.mp3")

    # Validate and download tracks
    tracklist_df = validate_tracklist(mix_id, tracklist_csv)
    if tracklist_df is not None:
        for _, track in tracklist_df.iterrows():
            artist = str(track['artist'])  # Ensure string
            title = str(track['title'])   # Ensure string
            try:
                i_track = int(track['i_track'])  # Convert to int safely
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid i_track value for track {artist} - {title}: {e}. Skipping.", file=sys.stderr)
                continue
            
            # Skip unidentified tracks
            if 'ID' == title.upper() or 'ID' == artist.upper():
                print(f"Warning: Skipping unidentified track: {artist} - {title}", file=sys.stderr)
                continue

            search_query = f"{artist} {title}"
            print(f"Searching for track: {search_query}")
            if not download_audio(search_query, output_dir, str(i_track), mix_id):
                print(f"Warning: Failed to download track '{search_query}'", file=sys.stderr)
            else:
                print(f"Successfully downloaded track as ../data/mp3/{mix_id}/{i_track}.mp3")

if __name__ == "__main__":
    main()