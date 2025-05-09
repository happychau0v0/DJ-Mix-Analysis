import os
import sys
import pandas as pd
from bs4 import BeautifulSoup
from database import *

def parse_timestamp(timestamp_str):
    if not timestamp_str:
        return 0
    parts = timestamp_str.strip().split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return 0

def scrape_tracklist(html_file):
    print(f"Parsing tracklist from HTML file: {html_file}")
    try:
        with open(html_file, 'r', encoding='utf-8') as file:
            html_content = file.read()
    except Exception as e:
        print(f"Error reading HTML file: {e}", file=sys.stderr)
        sys.exit(1)

    soup = BeautifulSoup(html_content, 'html.parser')
    mix_title = soup.select_one('meta[property="og:title"]')['content'] if soup.select_one('meta[property="og:title"]') else "Unknown Mix"
    print(f"Mix title: {mix_title}")

    tracks = []
    tl_tab = soup.find(id='tlTab')
    if not tl_tab:
        print("Warning: No tracks found. Invalid HTML structure.", file=sys.stderr)
        return tracks, mix_title, None

    mixId = tl_tab.attrs.get('data-id')
    if not mixId:
        print("Mix ID not found, skipping.", file=sys.stderr)
        return tracks, mix_title, None

    for i, track_element in enumerate(tl_tab.find_all('div', class_='tlpItem')):
        try:
            dataId = track_element.attrs.get('data-id')
            if not dataId:
                print(f"Track {i:02}: No data-id found, skipping.")
                continue

            timestamp = track_element.find('div', id=f'cue_{dataId}').text
            title = track_element.find('meta', itemprop='name')['content']
            artists = track_element.find('meta', itemprop='byArtist')['content']

            tracks.append({
                'mix_id': mixId,
                'i_track': i,
                'track_id': dataId,
                'timestamp': timestamp,
                'artist': artists,
                'title': title
            })
        except Exception as e:
            print(f"Warning: Error parsing track {i}: {e}", file=sys.stderr)

    print(f"Successfully parsed {len(tracks)} tracks")
    return tracks, mix_title, mixId

def run_scraper(html_file_base, output_csv_path=None, mixes_db_path='../data/mixes.csv'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_full_path = os.path.join(script_dir, '..', 'tracklists', html_file_base + '.html')

    try:
        tracks, mix_title, mixId = scrape_tracklist(html_full_path)
    except Exception as e:
        print(f"Error during scraping: {e}", file=sys.stderr)
        return None, None

    if not tracks or not mixId:
        print(f"No tracks found or mixId missing for {html_file_base}.", file=sys.stderr)
        return None, None

    output_csv_full_path = output_csv_path or os.path.join(script_dir, '..', 'data', 'meta', f"{mixId}.csv")
    save_to_csv(tracks, output_csv_full_path)
    update_mixes_database(mix_title, mixId, mixes_db_path)

    return mixId, output_csv_full_path