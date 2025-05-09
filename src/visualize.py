import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import numpy as np
import librosa

COLORS = ['red', 'blue', 'green', 'purple', 'orange']
MATCH_RATE_THRESHOLD = 0.5

def get_mix_beats(mix_id):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mix_wav_path = os.path.join(script_dir, '..', 'data', 'wav', mix_id, 'mix.wav')
    if not os.path.exists(mix_wav_path):
        print(f"Mix WAV file not found: {mix_wav_path}")
        return None
    y, sr = librosa.load(mix_wav_path)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    return beats

def timestamp_to_seconds(timestamp):
    time_parts = timestamp.split(':')
    if len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(time_parts) == 2:
        minutes, seconds = map(int, time_parts)
        return minutes * 60 + seconds
    raise ValueError(f"Invalid timestamp format: {timestamp}")

def visualize_alignment(mix_id, pkl_path, viz_dir):
    try:
        results_df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"Error reading .pkl file at {pkl_path}: {e}")
        return
    if results_df.empty:
        print(f"No results to visualize for mix {mix_id}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(script_dir, '..', 'data', 'meta', f"{mix_id}.csv")
    try:
        gt_df = pd.read_csv(gt_path)
    except Exception as e:
        print(f"Error reading ground truth CSV at {gt_path}: {e}")
        gt_df = pd.DataFrame()

    xmax = max(results_df['mix_cue_out_beat'].max(), results_df['mix_cue_in_beat'].max())
    ymax = max(results_df['track_cue_out_beat'].max(), results_df['track_cue_in_beat'].max())
    plt.figure(figsize=(int(xmax / ymax * 2), 4))
    cnt = 0

    for i, (_, track) in enumerate(results_df.iterrows()):
        print(i, track['mix_cue_in_time'], track['mix_cue_out_time'], track['match_rate'])
        if track['mix_cue_out_time'] - track['mix_cue_in_time'] < 20 or track['match_rate'] < 0.25:
            continue

        color = COLORS[i % len(COLORS)]
        wp = track['wp']
        plt.plot(wp[:, 1], wp[:, 0], color=color)
        textoffset = 5
        plt.text(
            wp[0, 1] + textoffset, wp[0, 0] + textoffset, f"{track['i_track']}",
            color='white', bbox={'facecolor': color, 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
            verticalalignment='bottom', horizontalalignment='left'
        )
        plt.axvline(track['mix_cue_in_beat'], color=color, linestyle='--', linewidth=1)
        plt.axvline(track['mix_cue_out_beat'], color=color, linestyle='--', linewidth=1)
        plt.text(
            track['mix_cue_in_beat'] + textoffset, ymax * 1.10 + textoffset, f"{track['i_track']}_in",
            color='white', bbox={'facecolor': color, 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
            verticalalignment='bottom', horizontalalignment='left'
        )
        plt.text(
            track['mix_cue_out_beat'] + textoffset, ymax * 1.10 + textoffset, f"{track['i_track']}_out",
            color='white', bbox={'facecolor': color, 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
            verticalalignment='bottom', horizontalalignment='left'
        )
        if track['match_rate'] > MATCH_RATE_THRESHOLD:
            # print(i, cnt, track['match_rate'])
            box_height = 0.04 * ymax
            plt.gca().add_patch(mpatches.Rectangle(
                xy=(track['mix_cue_in_beat'], box_height * cnt), width=track['mix_cue_out_beat'] - track['mix_cue_in_beat'],
                height=box_height, linewidth=0, facecolor=color, alpha=0.5
            ))

        cnt += 1

    mix_beats = get_mix_beats(mix_id)
    if not gt_df.empty and mix_beats is not None:
        for _, gt_row in gt_df.iterrows():
            i_track = gt_row['i_track']
            timestamp = gt_row['timestamp']
            if pd.notna(timestamp):
                try:
                    timestamp_secs = timestamp_to_seconds(timestamp)
                    i_beat = np.argmin(np.abs(mix_beats - timestamp_secs))
                    plt.axvline(i_beat, color='black', linestyle='-', linewidth=1.5)
                    plt.text(
                        i_beat + textoffset, ymax * 1.05, f"GT_{i_track}",
                        color='white', bbox={'facecolor': 'black', 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
                        verticalalignment='bottom', horizontalalignment='left'
                    )
                except ValueError as e:
                    print(f"Invalid timestamp format for track {i_track}: {timestamp} - {e}")

    plt.xlim(0, xmax * 1.03)
    plt.ylim(0, ymax * 1.17)
    plt.legend(handles=[	mpatches.Patch(color='black', label='Ground Truth')],
        bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode='expand', borderaxespad=0.)
    plt.ylabel('Track beat frame')
    plt.xlabel('Mix beat frame')
    plt.tight_layout()

    viz_path = os.path.join(viz_dir, f"{mix_id}.pdf")
    plt.savefig(viz_path)
    plt.close()
    print(f"Saved visualization to {viz_path}")
    return viz_path

def run_visualizer(mix_id, pkl_path, viz_base_dir='../data/dtwviz'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_viz_dir = os.path.join(script_dir, viz_base_dir)

    if not os.path.exists(pkl_path):
        print(f"Alignment .pkl file not found at {pkl_path}", file=sys.stderr)
        return None

    try:
        os.makedirs(full_viz_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating visualization directory {full_viz_dir}: {e}", file=sys.stderr)
        return None

    try:
        return visualize_alignment(mix_id, pkl_path, full_viz_dir)
    except Exception as e:
        print(f"Error during visualization for mix {mix_id}: {e}", file=sys.stderr)
        return None