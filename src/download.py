import os
import sys
import pandas as pd
import yt_dlp
from rapidfuzz import fuzz
from urllib.parse import urlparse

def validate_mix_id(mix_id, mixes_csv):
    df = pd.read_csv(mixes_csv)
    if mix_id in df['mix_id'].values:
        return df[df['mix_id'] == mix_id]['mix_title'].iloc[0]
    print(f"Error: mix_id {mix_id} not found in {mixes_csv}", file=sys.stderr)
    sys.exit(1)

def validate_tracklist(mix_id, tracklist_csv):
    try:
        df = pd.read_csv(tracklist_csv)
        return df if not df.empty else None
    except Exception as e:
        print(f"Error reading {tracklist_csv}: {e}", file=sys.stderr)
        return None

def is_likely_match(search_query, video_title, threshold=40):
    score = fuzz.partial_ratio(search_query.lower(), video_title.lower())
    print('Likelihood: ', score)
    return score >= threshold

def extract_youtube_id(url):
    parsed = urlparse(url)
    if parsed.netloc in ('www.youtube.com', 'youtube.com', 'youtu.be'):
        return parsed.path[1:] if parsed.netloc == 'youtu.be' else next((param[2:] for param in parsed.query.split('&') if param.startswith('v=')), None)
    return None

def download_audio(search_query, output_path, filename, mix_id):
    output_dir = os.path.join('../data/mp3', str(mix_id))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{filename}.mp3")
    
    if os.path.exists(output_file):
        print(f"Info: File already exists at {output_file}. Skipping download.", file=sys.stderr)
        return True
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, f'{filename}.%(ext)s'),
        'noplaylist': True,
        'default_search': 'ytsearch',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch:{search_query}", download=False)
            if not info or 'entries' not in info or not info['entries']:
                print(f"Warning: No results found for '{search_query}'", file=sys.stderr)
                return False

            video = info['entries'][0]
            video_title = video.get('title', '')
            video_url = video.get('webpage_url', '')

            if not is_likely_match(search_query, video_title):
                print(f"Warning: Top result '{video_title}' for '{search_query}' seems unrelated. Skipping.", file=sys.stderr)
                return False

            print(f"Downloading '{video_title}' for '{search_query}' as {filename}.mp3")
            ydl.download([video_url])
            return True
    except Exception as e:
        print(f"Error downloading '{search_query}': {e}", file=sys.stderr)
        return False

def run_downloader(mix_id, mixes_csv_path='../data/mixes.csv', meta_dir='../data/meta', mp3_base_dir='../data/mp3'):
    mix_id = str(mix_id)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_mixes_csv_path = os.path.join(script_dir, mixes_csv_path)
    full_tracklist_csv_path = os.path.join(script_dir, meta_dir, f"{mix_id}.csv")
    full_output_dir = os.path.join(script_dir, mp3_base_dir, mix_id)

    try:
        mix_title = validate_mix_id(mix_id, full_mixes_csv_path)
    except SystemExit:
        return None

    print(f"--- Downloading Mix: {mix_title} ---")
    mix_downloaded = download_audio(mix_title, full_output_dir, "mix", mix_id)
    if not mix_downloaded:
        print(f"Warning: Failed to download original mix '{mix_title}'", file=sys.stderr)
    else:
        print(f"Successfully processed original mix.")

    print(f"--- Downloading Tracks for Mix ID: {mix_id} ---")
    tracklist_df = validate_tracklist(mix_id, full_tracklist_csv_path)
    if tracklist_df is None:
        print(f"Warning: Could not read or validate tracklist at {full_tracklist_csv_path}. Skipping track downloads.", file=sys.stderr)
    else:
        tracks_downloaded_count = 0
        tracks_failed_count = 0
        for _, track in tracklist_df.iterrows():
            artist, title = str(track['artist']), str(track['title'])
            try:
                i_track = int(track['i_track'])
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid i_track value for track {artist} - {title}: {e}. Skipping.", file=sys.stderr,)
                tracks_failed_count += 1
                continue

            if 'ID' in (artist.upper(), title.upper()):
                print(f"Info: Skipping unidentified track: {artist} - {title}", file=sys.stderr)
                continue

            search_query = f"{artist} {title}"
            print(f"Processing track {i_track}: {search_query}")
            if download_audio(search_query, full_output_dir, str(i_track), mix_id):
                print(f"Successfully processed track {i_track}.")
                tracks_downloaded_count += 1
            else:
                print(f"Warning: Failed to download track '{search_query}'", file=sys.stderr)
                tracks_failed_count += 1
        print(f"--- Track Download Summary: {tracks_downloaded_count} processed, {tracks_failed_count} failed/skipped ---")

    return full_output_dir
