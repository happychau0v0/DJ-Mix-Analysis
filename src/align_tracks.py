#!/usr/bin/env python3
"""
DJ Mix Track Alignment

This script aligns individual tracks to the original mix using Dynamic Time Warping (DTW).
It extracts audio features, performs DTW alignment, identifies cue points, and saves the results.

Usage:
    python align_tracks.py [mix_id] [--features chroma,mfcc] [--key-invariant]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import namedtuple
import joblib
from pydub import AudioSegment
import multiprocessing as mp
from functools import partial

# Constants
SR = 22050  # Sample rate
CACHE_DIR = './cache'
MATCH_RATE_THRESHOLD = 0.4
COLORS = ['C0', 'C1', 'C2', 'C3', 'C4']

# Define feature combinations
Case = namedtuple('Case', ['features', 'key_invariant'])
CASES = [
    Case(features=['mfcc'], key_invariant=False),
    Case(features=['chroma'], key_invariant=False),
    Case(features=['chroma'], key_invariant=True),
    Case(features=['chroma', 'mfcc'], key_invariant=False),
    Case(features=['chroma', 'mfcc'], key_invariant=True),
]

# Initialize cache
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=1)

# Feature extraction functions
def beats(path):
    """Extract beat positions from audio file."""
    y, sr = librosa.load(path)
    tempo, beats_ = librosa.beat.beat_track(y=y, sr=sr, units='time')
    return beats_

def mfcc(path):
    """Extract MFCC features from audio file."""
    sig_, sr = librosa.load(path)
    mfcc_ = librosa.feature.mfcc(y=sig_, sr=sr, n_mfcc=12)
    return mfcc_

def beat_mfcc(path):
    """Extract beat-synchronized MFCC features."""
    beats_ = beats(path)
    mfcc_ = mfcc(path)
    beat_mfcc_ = beat_aggregate(mfcc_, beats_)
    return beat_mfcc_

def chroma_cens(path):
    """Extract chroma CENS features from audio file."""
    sig, sr = librosa.load(path)
    chroma_cens_ = librosa.feature.chroma_cens(y=sig, sr=sr)
    return chroma_cens_

def beat_chroma_cens(path):
    """Extract beat-synchronized chroma CENS features."""
    beats_ = beats(path)
    chroma_cens_ = chroma_cens(path)
    beat_chroma_cens_ = beat_aggregate(chroma_cens_, beats_)
    return beat_chroma_cens_

def beat_aggregate(feature, beats, frames_per_beat=None):
    """Aggregate features by beat."""
    max_frame = feature.shape[1]
    beat_frames = librosa.time_to_frames(beats)
    beat_frames = beat_frames[beat_frames < max_frame]
    beat_feature = np.split(feature, beat_frames, axis=1)
    beat_feature = beat_feature[1:-1]  # only use features between beats
    beat_feature = [f.mean(axis=1) for f in beat_feature]
    beat_feature = np.array(beat_feature).T
    return beat_feature

def extract_features(path, feature_names):
    """Extract and combine multiple feature types."""
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
    combined_feature = np.concatenate(combined_feature, axis=0)
    return combined_feature

# Cue point detection
def find_cue(wp, cue_in=False, num_diag=32):
    """
    Find cue points in the warping path.
    
    Args:
        wp: Warping path from DTW
        cue_in: If True, find cue-in point; else cue-out
        num_diag: Number of diagonal steps to look for
        
    Returns:
        (cue point in beats on mix, cue point in beats on track)
    """
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

# DTW alignment (modified)
def align_track_to_mix(mix_feature, mix_beats, track_path, features=['chroma', 'mfcc'], key_invariant=True):
    """
    Align a track to a mix using DTW with precomputed mix features.
    
    Args:
        mix_feature: Precomputed mix features
        mix_beats: Precomputed mix beat positions
        track_path: Path to track audio file
        features: List of feature types
        key_invariant: Whether to use key-invariant matching
        
    Returns:
        Dictionary with alignment results
    """
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
    
    x = best_wp[:, 1][::-1]
    y = best_wp[:, 0][::-1]
    dx = np.diff(x)
    dy = np.diff(y)
    valid = dx != 0
    if valid.any():
        dydx = np.zeros_like(dx, dtype=float)
        dydx[valid] = dy[valid] / dx[valid]
        match_rate = (dydx == 1).sum() / len(dydx)
    else:
        match_rate = 0.0
    
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
    """Convert MP3 file to WAV format."""
    if os.path.exists(wav_path):
        print(f"WAV file already exists: {wav_path}")
        return
    print(f"Converting {mp3_path} to {wav_path}")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Worker function for parallel processing
def align_worker(mix_feature, mix_beats, features, key_invariant, track_data):
    """
    Worker function to align a single track in parallel.
    
    Args:
        mix_feature: Precomputed mix features
        mix_beats: Precomputed mix beats
        features: List of feature types
        key_invariant: Whether to use key-invariant matching
        track_data: Tuple of (track_wav_path, track_info)
        
    Returns:
        Dictionary with alignment results and track info
    """
    track_wav_path, track_info = track_data
    alignment_result = align_track_to_mix(mix_feature, mix_beats, track_wav_path, features, key_invariant)
    result = {
        'mix_id': track_info['mix_id'],
        'track_id': track_info['track_id'],
        'i_track': track_info['i_track'],
        'artist': track_info['artist'],
        'title': track_info['title'],
        'feature': '+'.join(features),
        'key_invariant': key_invariant,
        **alignment_result
    }
    return result

def process_mix(mix_id, features=['chroma', 'mfcc'], key_invariant=True):
    """
    Process a complete mix with all its tracks using multiple cores.
    
    Args:
        mix_id: ID of the mix to process
        features: List of feature types to use
        key_invariant: Whether to use key-invariant matching
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    mp3_dir = os.path.join(script_dir, '..', 'data', 'mp3', mix_id)
    wav_dir = os.path.join(script_dir, '..', 'data', 'wav', mix_id)
    meta_path = os.path.join(script_dir, '..', 'data', 'meta', f"{mix_id}.csv")
    results_dir = os.path.join(script_dir, '..', 'data', 'align')
    
    # Create directories
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if results already exist
    feature_id = '+'.join(features)
    result_path = os.path.join(results_dir, f"{mix_id}-{feature_id}")
    if key_invariant:
        result_path += '-keyinv'
    result_path += '.pkl'
    if os.path.exists(result_path):
        print(f"Results already exist: {result_path}")
        return result_path

    # Read tracklist
    try:
        tracklist = pd.read_csv(meta_path)
    except Exception as e:
        print(f"Error reading tracklist: {e}", file=sys.stderr)
        return
    
    # Convert mix MP3 to WAV
    mix_mp3_path = os.path.join(mp3_dir, "mix.mp3")
    mix_wav_path = os.path.join(wav_dir, "mix.wav")
    if not os.path.exists(mix_mp3_path):
        print(f"Mix MP3 file not found: {mix_mp3_path}", file=sys.stderr)
        return
    convert_mp3_to_wav(mix_mp3_path, mix_wav_path)
    
    # Prepare track data and convert MP3 to WAV
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
    
    # Compute mix features once
    mix_beats = beats(mix_wav_path)
    mix_feature = extract_features(mix_wav_path, features)
    
    # Parallel alignment using multiple cores
    num_cores = mp.cpu_count()  # Use all available cores
    with mp.Pool(processes=num_cores) as pool:
        worker = partial(align_worker, mix_feature, mix_beats, features, key_invariant)
        results = pool.map(worker, track_data_list)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_pickle(result_path)
    print(f"Saved alignment results to {result_path}")
    
    # Return the path to the results file
    return result_path

# Example usage (if run directly, though intended as module):
# if __name__ == "__main__":
#     test_mix_id = '6qdzkf9' # Replace with a valid mix ID
#     features_to_use = ['chroma', 'mfcc']
#     key_inv = True
#     pkl_file = process_mix(test_mix_id, features=features_to_use, key_invariant=key_inv)
#     if pkl_file and os.path.exists(pkl_file):
#         print(f"Alignment process finished. Results saved to: {pkl_file}")
#         # Optionally load and inspect results
#         # results = pd.read_pickle(pkl_file)
#         # print(results.head())
#     else:
#         print(f"Alignment process failed or results file not found for mix ID: {test_mix_id}")
