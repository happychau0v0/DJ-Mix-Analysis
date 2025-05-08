import os
import sys
import pandas as pd
import numpy as np
import librosa
from collections import namedtuple
import joblib
from pydub import AudioSegment
import multiprocessing as mp
from functools import partial
from feature_extraction import *

SR = 22050
CACHE_DIR = './cache'
MATCH_RATE_THRESHOLD = 0.5

Case = namedtuple('Case', ['features', 'key_invariant'])
CASES = [
    Case(features=['mfcc'], key_invariant=False),
    Case(features=['chroma'], key_invariant=False),
    Case(features=['chroma'], key_invariant=True),
    Case(features=['chroma', 'mfcc'], key_invariant=False),
    Case(features=['chroma', 'mfcc'], key_invariant=True),
]

os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=1)

def extract_features(path, feature_names):
    combined_feature = []
    for feature_name in feature_names:
        if feature_name == 'chroma':
            f = beat_chroma_cens(path).astype('float32')
        elif feature_name == 'mfcc':
            f = beat_mfcc(path).astype('float32')
        else:
            raise Exception(f'Unknown feature: {feature_name}')
        f = (f - f.mean()) / f.std()
        combined_feature.append(f)
    return np.concatenate(combined_feature, axis=0)

def find_cue(wp, cue_in=False, num_diag=32):
    if num_diag == 0:
        if cue_in:
            return wp[-1, 1], wp[-1, 0]
        else:
            return wp[0, 1], wp[0, 0]

    x, y = wp[::-1, 1], wp[::-1, 0]
    dx, dy = np.diff(x), np.diff(y)

    with np.errstate(divide='ignore'):
        slope = dy / dx
    slope[np.isinf(slope)] = 0

    if cue_in:
        slope = slope[::-1].cumsum()
        slope[num_diag:] = slope[num_diag:] - slope[:-num_diag]
        slope = slope[::-1]
        i_diag = np.nonzero(slope == num_diag)[0]
        if len(i_diag) == 0:
            return find_cue(wp, cue_in, num_diag // 2)
        else:
            i = i_diag[0]
            return x[i], y[i]
    else:
        slope = slope.cumsum()
        slope[num_diag:] = slope[num_diag:] - slope[:-num_diag]
        i_diag = np.nonzero(slope == num_diag)[0]
        if len(i_diag) == 0:
            return find_cue(wp, cue_in, num_diag // 2)
        else:
            i = i_diag[-1]
        return x[i] + 1, y[i] + 1

def align_track_to_mix(mix_feature, mix_beats, track_path, features=['chroma', 'mfcc'], key_invariant=True):
    track_feature = extract_features(track_path, features)
    track_beats = beats(track_path)
    
    pitch_shifts = np.arange(12) if key_invariant else [0]
    best_cost = np.inf
    best_key_change = np.nan
    best_wp = None
    costs = []
    
    for pitch_shift in pitch_shifts:
        if pitch_shift == 0:
            X, Y = track_feature, mix_feature
        else:
            X, Y = track_feature.copy(), mix_feature.copy()
            X[:12] = np.roll(X[:12], pitch_shift, axis=0)
        
        D, wp = librosa.sequence.dtw(X, Y, subseq=True)
        matching_function = D[-1, :] / wp.shape[0]
        cost = matching_function.min()
        costs.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_key_change = pitch_shift
            best_wp = wp
    
    x, y = best_wp[::-1, 1], best_wp[::-1, 0]
    dx, dy = np.diff(x), np.diff(y)
    valid = dx != 0
    match_rate = ((dy[valid] / dx[valid]) == 1).sum() / len(dx) if valid.any() else 0.0
    
    mix_cue_in_beat, track_cue_in_beat = find_cue(best_wp, cue_in=True)
    mix_cue_out_beat, track_cue_out_beat = find_cue(best_wp, cue_in=False)
    mix_cue_in_time = mix_beats[int(mix_cue_in_beat)] if int(mix_cue_in_beat) < len(mix_beats) else mix_beats[-1]
    track_cue_in_time = track_beats[int(track_cue_in_beat)] if int(track_cue_in_beat) < len(track_beats) else track_beats[-1]
    mix_cue_out_time = mix_beats[int(mix_cue_out_beat)] if int(mix_cue_out_beat) < len(mix_beats) else mix_beats[-1]
    track_cue_out_time = track_beats[int(track_cue_out_beat)] if int(track_cue_out_beat) < len(track_beats) else track_beats[-1]
    
    return {
        'match_rate': match_rate,
        'key_change': best_key_change,
        'best_cost': best_cost,
        'costs': costs,
        'wp': best_wp,
        'mix_cue_in_time': mix_cue_in_time,
        'mix_cue_out_time': mix_cue_out_time,
        'track_cue_in_time': track_cue_in_time,
        'track_cue_out_time': track_cue_out_time,
        'mix_cue_in_beat': mix_cue_in_beat,
        'mix_cue_out_beat': mix_cue_out_beat,
        'track_cue_in_beat': track_cue_in_beat,
        'track_cue_out_beat': track_cue_out_beat,
    }

def convert_mp3_to_wav(mp3_path, wav_path):
    if os.path.exists(wav_path):
        print(f"WAV file already exists: {wav_path}")
        return
    print(f"Converting {mp3_path} to {wav_path}")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def align_worker(mix_feature, mix_beats, features, key_invariant, track_data):
    track_wav_path, track_info = track_data
    alignment_result = align_track_to_mix(mix_feature, mix_beats, track_wav_path, features, key_invariant)
    return {
        'mix_id': track_info['mix_id'],
        'track_id': track_info['track_id'],
        'i_track': track_info['i_track'],
        'artist': track_info['artist'],
        'title': track_info['title'],
        'feature': '+'.join(features),
        'key_invariant': key_invariant,
        **alignment_result
    }

def process_mix(mix_id, features=['chroma', 'mfcc'], key_invariant=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mp3_dir = os.path.join(script_dir, '..', 'data', 'mp3', mix_id)
    wav_dir = os.path.join(script_dir, '..', 'data', 'wav', mix_id)
    meta_path = os.path.join(script_dir, '..', 'data', 'meta', f"{mix_id}.csv")
    results_dir = os.path.join(script_dir, '..', 'data', 'align')
    
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    feature_id = '+'.join(features)
    result_path = os.path.join(results_dir, f"{mix_id}-{feature_id}{'-keyinv' if key_invariant else ''}.pkl")
    if os.path.exists(result_path):
        print(f"Results already exist: {result_path}")
        return result_path

    try:
        tracklist = pd.read_csv(meta_path)
    except Exception as e:
        print(f"Error reading tracklist: {e}", file=sys.stderr)
        return
    
    mix_mp3_path = os.path.join(mp3_dir, "mix.mp3")
    mix_wav_path = os.path.join(wav_dir, "mix.wav")
    if not os.path.exists(mix_mp3_path):
        print(f"Mix MP3 file not found: {mix_mp3_path}", file=sys.stderr)
        return
    convert_mp3_to_wav(mix_mp3_path, mix_wav_path)
    
    track_data_list = []
    for _, track in tracklist.iterrows():
        i_track = track['i_track']
        track_mp3_path = os.path.join(mp3_dir, f"{i_track}.mp3")
        track_wav_path = os.path.join(wav_dir, f"{i_track}.wav")
        if not os.path.exists(track_mp3_path):
            print(f"Track MP3 file not found: {track_mp3_path}", file=sys.stderr)
            continue
        convert_mp3_to_wav(track_mp3_path, track_wav_path)
        if not os.path.exists(track_wav_path):
            print(f"Failed to convert track to WAV: {track_wav_path}", file=sys.stderr)
            continue
        track_info = {
            'mix_id': mix_id,
            'track_id': track['track_id'],
            'i_track': i_track,
            'artist': track['artist'],
            'title': track['title']
        }
        track_data_list.append((track_wav_path, track_info))
    
    print("Extracting mix features...")
    mix_beats = beats(mix_wav_path)
    mix_feature = extract_features(mix_wav_path, features)
    
    print("Performing tracks alignment...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        worker = partial(align_worker, mix_feature, mix_beats, features, key_invariant)
        results = pool.map(worker, track_data_list)
    
    results_df = pd.DataFrame(results)
    results_df.to_pickle(result_path)
    print(f"Saved alignment results to {result_path}")
    return result_path