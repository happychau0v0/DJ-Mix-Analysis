import asyncio
import os
import base64
import shutil
from shazamio import Shazam
from pydub import AudioSegment
import logging
from database import save_to_csv, update_mixes_database

def generate_mix_id(mix_name: str) -> str:
    encoded = base64.b64encode(mix_name.encode()).decode()
    return encoded[:8]

async def detect_songs_in_mix(mix_path: str, interval_seconds: int = 60, segment_duration: int = 15) -> tuple:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    mix_name = os.path.splitext(os.path.basename(mix_path))[0]
    mix_id = generate_mix_id(mix_name)
    shazam = Shazam()
    
    if not os.path.exists(mix_path):
        logger.error(f"Mix file not found: {mix_path}")
        return mix_id, None
    
    destination_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'mp3', mix_id)
    destination_path = os.path.join(destination_dir, 'mix.mp3')
    try:
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copy2(mix_path, destination_path)
        logger.info(f"Copied mix file to {destination_path}")
    except Exception as e:
        logger.error(f"Error copying mix file: {e}")
        return mix_id, None
    
    try:
        audio = AudioSegment.from_mp3(mix_path)
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        return mix_id, None
    
    mix_duration_ms = len(audio)
    detected_songs = []
    current_time_ms = 0
    i_track = 0
    
    while current_time_ms < mix_duration_ms:
        segment = audio[current_time_ms:current_time_ms + (segment_duration * 1000)]
        temp_file = f"temp_segment_{current_time_ms}.mp3"
        segment.export(temp_file, format="mp3")
        
        try:
            result = await shazam.recognize(temp_file)
            if result and "track" in result:
                track = result["track"]
                song_info = {
                    "mix_id": mix_id,
                    "i_track": i_track,
                    "track_id": "",
                    "timestamp": "",
                    "artist": track.get("subtitle", ""),
                    "title": track.get("title", "")
                }
                # Check for duplicates based on artist and title
                if not any(song["artist"] == song_info["artist"] and song["title"] == song_info["title"] for song in detected_songs):
                    detected_songs.append(song_info)
                    logger.info(f"Detected: {song_info['title']} by {song_info['artist']} at {song_info['timestamp']}")
                    i_track += 1
            else:
                logger.info(f"No song detected at {current_time_ms/1000}s")
        except Exception as e:
            logger.error(f"Error processing segment at {current_time_ms/1000}s: {e}")
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        current_time_ms += interval_seconds * 1000
    
    # Reassign i_track sequentially after deduplication
    for index, song in enumerate(detected_songs):
        song["i_track"] = index
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv_path = os.path.join(script_dir, '..', 'data', 'meta', f"{mix_id}.csv")
    if detected_songs:
        save_to_csv(detected_songs, output_csv_path)
    
    update_mixes_database(mix_name, mix_id, '../data/mixes.csv')
    
    return mix_id, output_csv_path

if __name__ == "__main__":
    async def main():
        mix_id, csv_path = await detect_songs_in_mix("../data/mix/test/johnsummit.mp3", interval_seconds=600, segment_duration=15)
        print(f"Mix ID: {mix_id}")
        print(f"Output CSV: {csv_path}")
    
    asyncio.run(main())