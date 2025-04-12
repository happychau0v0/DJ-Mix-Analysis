#!/usr/bin/env python3
"""
1001Tracklists HTML Parser

This script parses tracklist information from a saved HTML file of a 1001tracklists.com page and saves it as a CSV file.
"""

import argparse
import csv
import re
import sys
import os
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

def parse_timestamp(timestamp_str):
    """Convert timestamp from MM:SS format to seconds."""
    if not timestamp_str or timestamp_str == "":
        return 0
    
    # Handle different timestamp formats
    if ':' in timestamp_str:
        parts = timestamp_str.strip().split(':')
        if len(parts) == 2:  # MM:SS
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    
    # Return 0 if no valid format
    return 0

def scrape_tracklist(html_file):
    """Parse tracklist information from a saved HTML file of 1001tracklists.com."""
    print(f"Parsing tracklist from HTML file: {html_file}")
    
    try:
        with open(html_file, 'r', encoding='utf-8') as file:
            html_content = file.read()
    except Exception as e:
        print(f"Error reading the HTML file: {e}", file=sys.stderr)
        sys.exit(1)

    soup = BeautifulSoup(html_content, 'html.parser')
    
    try:
        title_element = soup.select_one('meta[property="og:title"]')
        mix_title = title_element['content'] if title_element else "Unknown Mix"
        print(f"Mix title: {mix_title}")
    except Exception as e:
        print(f"Warning: Could not extract mix title: {e}", file=sys.stderr)
        mix_title = "Unknown Mix"
    
    tracks = []
    tl_tab = soup.find(id='tlTab')
    
    if not tl_tab:
        print("Warning: No tracks found. The HTML structure might not contain the tracklist, or the file might be invalid.", file=sys.stderr)
        return tracks, mix_title
    
    mixId = tl_tab.attrs.get('data-id')
    print(mixId)
    if mixId is None:
        print(f"Mix id not found, skipping.")
        return tracks, mix_title
    
    track_elements = tl_tab.find_all('div', class_='tlpItem')
    
    if not track_elements:
        print("Warning: No track elements (<div class='tlpItem'>) found inside tlTab. The HTML structure might be different.", file=sys.stderr)
    
    for i, track_element in enumerate(track_elements):
        try:
            track_num = f"{i:02}"
            dataId = track_element.attrs.get('data-id')
            if dataId is None:
                print(f"Track {track_num}: No data-id found, skipping.")
                continue
            
            print(f"Track {track_num} data-id: {dataId}")
            timestamp_seconds = track_element.find('div', id=f'cue_{dataId}').text
            print(timestamp_seconds)
            
            title_meta = track_element.find('meta', itemprop='name')
            title = title_meta['content']
            artists_meta = track_element.find('meta', itemprop='byArtist')
            artists = artists_meta['content']
            
            tracks.append({
                'mix_id': mixId,
                'i_track': i,
                'track_id': dataId,
                'timestamp': timestamp_seconds,
                'artist': artists,
                'title': title
            })
            
        except Exception as e:
            print(f"Warning: Error parsing track {i}: {e}", file=sys.stderr)
    
    print(f"Successfully parsed {len(tracks)} tracks")
    return tracks, mix_title, mixId  # Modified to return mixId

def save_to_csv(tracks, output_file):
    """Save tracklist to CSV file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df = pd.DataFrame(tracks)
        df.to_csv(output_file, index=False)
        print(f"Tracklist saved to {output_file}")
    except Exception as e:
        print(f"Error saving to CSV: {e}", file=sys.stderr)
        sys.exit(1)

def update_mixes_database(mix_title, mixId, database_path):
    """Update mixes.csv with new mix entry."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        database_full_path = os.path.join(script_dir, database_path)
        
        os.makedirs(os.path.dirname(database_full_path), exist_ok=True)
        
        # Create new entry
        new_entry = {'mix_id': mixId, 'mix_title': mix_title}
        
        # Check if database exists
        if os.path.exists(database_full_path):
            df = pd.read_csv(database_full_path)
            # Check if mix_id already exists
            if mixId not in df['mix_id'].values:
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(database_full_path, index=False)
                print(f"Updated mixes database with mix ID {mixId}")
            else:
                print(f"Mix ID {mixId} already exists in database")
        else:
            # Create new database
            pd.DataFrame([new_entry]).to_csv(database_full_path, index=False)
            print(f"Created new mixes database with mix ID {mixId}")
            
    except Exception as e:
        print(f"Error updating mixes database: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Parse tracklist from a saved 1001tracklists.com HTML file')
    parser.add_argument('file', help='Path to the saved HTML file of the 1001tracklists.com tracklist page')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    args = parser.parse_args()
    
    # Parse tracklist
    tracks, mix_title, mixId = scrape_tracklist(args.file)  # Modified to receive mixId
    
    if not tracks:
        print("No tracks found. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Determine output filename
    if not args.output:
        output_file = f"{mixId}.csv"  # Modified to use mixId
    else:
        output_file = args.output
    
    # Save tracklist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'data', 'meta', output_file)
    save_to_csv(tracks, output_dir)
    
    # Update mixes database
    mixes_db_path = '../data/mixes.csv'
    update_mixes_database(mix_title, mixId, mixes_db_path)

if __name__ == "__main__":
    main()