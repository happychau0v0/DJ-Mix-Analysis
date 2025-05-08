import os
import sys
import pandas as pd
from bs4 import BeautifulSoup

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

def save_to_csv(tracks, output_file):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pd.DataFrame(tracks).to_csv(output_file, index=False)
        print(f"Tracklist saved to {output_file}")
    except Exception as e:
        print(f"Error saving to CSV: {e}", file=sys.stderr)
        sys.exit(1)

def update_mixes_database(mix_title, mixId, database_path):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        database_full_path = os.path.join(script_dir, database_path)
        os.makedirs(os.path.dirname(database_full_path), exist_ok=True)
        
        new_entry = {'mix_id': mixId, 'mix_title': mix_title}
        if os.path.exists(database_full_path):
            df = pd.read_csv(database_full_path)
            if mixId not in df['mix_id'].values:
                pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True).to_csv(database_full_path, index=False)
                print(f"Updated mixes database with mix ID {mixId}")
            else:
                print(f"Mix ID {mixId} already exists in database")
        else:
            pd.DataFrame([new_entry]).to_csv(database_full_path, index=False)
            print(f"Created new mixes database with mix ID {mixId}")
    except Exception as e:
        print(f"Error updating mixes database: {e}", file=sys.stderr)

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