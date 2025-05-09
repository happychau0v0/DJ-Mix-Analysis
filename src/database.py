import os
import sys
import pandas as pd

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