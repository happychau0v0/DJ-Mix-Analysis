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

def seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def visualize_alignment(mix_id, pkl_path, viz_dir, meta_dir):
    try:
        results_df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"Error reading .pkl file at {pkl_path}: {e}")
        return
    if results_df.empty:
        print(f"No results to visualize for mix {mix_id}")
        return

    gt_path = os.path.join(meta_dir, f"{mix_id}.csv")
    try:
        gt_df = pd.read_csv(gt_path)
    except Exception as e:
        print(f"Error reading ground truth CSV at {gt_path}: {e}")
        gt_df = pd.DataFrame()

    if not gt_df.empty:
        track_info = gt_df.set_index('i_track')[['artist', 'title']].to_dict(orient='index')
    else:
        track_info = {}

    xmax = max(results_df['mix_cue_out_beat'].max(), results_df['mix_cue_in_beat'].max())
    ymax = max(results_df['track_cue_out_beat'].max(), results_df['track_cue_in_beat'].max())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(int(xmax / ymax * 2), 6), gridspec_kw={'height_ratios': [2, 2]})
    cnt = 0
    selected_tracks = []

    for i, (_, track) in enumerate(results_df.iterrows()):
        if track['mix_cue_out_time'] - track['mix_cue_in_time'] < 20 or track['match_rate'] < 0.25:
            continue

        color = COLORS[i % len(COLORS)]
        wp = track['wp']
        ax1.plot(wp[:, 1], wp[:, 0], color=color)
        textoffset = 5
        ax1.text(
            wp[0, 1] + textoffset, wp[0, 0] + textoffset, f"{track['i_track']}",
            color='white', bbox={'facecolor': color, 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
            verticalalignment='bottom', horizontalalignment='left'
        )
        ax1.axvline(track['mix_cue_in_beat'], color=color, linestyle='--', linewidth=1)
        ax1.axvline(track['mix_cue_out_beat'], color=color, linestyle='--', linewidth=1)
        ax1.text(
            track['mix_cue_in_beat'] + textoffset, ymax * 1.10 + textoffset, f"{track['i_track']}_in",
            color='white', bbox={'facecolor': color, 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
            verticalalignment='bottom', horizontalalignment='left'
        )
        ax1.text(
            track['mix_cue_out_beat'] + textoffset, ymax * 1.10 + textoffset, f"{track['i_track']}_out",
            color='white', bbox={'facecolor': color, 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
            verticalalignment='bottom', horizontalalignment='left'
        )
        if track['match_rate'] > MATCH_RATE_THRESHOLD:
            box_height = 0.04 * ymax
            ax1.add_patch(mpatches.Rectangle(
                xy=(track['mix_cue_in_beat'], box_height * cnt), width=track['mix_cue_out_beat'] - track['mix_cue_in_beat'],
                height=box_height, linewidth=0, facecolor=color, alpha=0.5
            ))
            i_track = track['i_track']
            artist = track_info.get(i_track, {}).get('artist', 'Unknown')
            title = track_info.get(i_track, {}).get('title', 'Unknown')
            mix_cue_in_time = seconds_to_timestamp(track['mix_cue_in_time'])
            mix_cue_out_time = seconds_to_timestamp(track['mix_cue_out_time'])
            match_rate = round(track['match_rate'], 3)
            selected_tracks.append({
                'i_track': i_track,
                'song': artist + ' - ' + title,
                'mix_cue_in_time': mix_cue_in_time,
                'mix_cue_out_time': mix_cue_out_time,
                'match_rate': match_rate
            })
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
                    ax1.axvline(i_beat, color='black', linestyle='-', linewidth=1.5)
                    ax1.text(
                        i_beat + textoffset, ymax * 1.05, f"GT_{i_track}",
                        color='white', bbox={'facecolor': 'black', 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
                        verticalalignment='bottom', horizontalalignment='left'
                    )
                except ValueError as e:
                    print(f"Invalid timestamp format for track {i_track}: {timestamp} - {e}")

    ax1.set_xlim(0, xmax * 1.03)
    ax1.set_ylim(0, ymax * 1.17)
    ax1.legend(handles=[mpatches.Patch(color='black', label='Ground Truth')],
               bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode='expand', borderaxespad=0.)
    ax1.set_ylabel('Track beat frame')
    ax1.set_xlabel('Mix beat frame')

    if selected_tracks:
        table_data = [[track['i_track'], track['song'], track['mix_cue_in_time'], track['mix_cue_out_time'], track['match_rate']] 
                      for track in selected_tracks]
        col_labels = ['Track Index', 'Song', 'Cue In Time', 'Cue Out Time', "Match Rate"]
        col_widths = [0.1, 0.6, 0.1, 0.1, 0.1]
        table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colWidths=col_widths)

        for i in range(len(table_data)): 
            table[(i + 1, 1)].set_text_props(horizontalalignment='left')
    ax2.axis('off')

    fig.tight_layout()
    viz_path = os.path.join(viz_dir, f"{mix_id}.pdf")
    fig.savefig(viz_path)
    plt.close(fig)
    print(f"Saved visualization to {viz_path}")
    return viz_path

def run_visualizer(mix_id, pkl_path, viz_base_dir='../data/dtwviz', meta_dir='../data/meta'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_viz_dir = os.path.join(script_dir, viz_base_dir)
    full_meta_dir = os.path.abspath(os.path.join(script_dir, meta_dir))

    if not os.path.exists(pkl_path):
        print(f"Alignment .pkl file not found at {pkl_path}", file=sys.stderr)
        return None

    try:
        os.makedirs(full_viz_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating visualization directory {full_viz_dir}: {e}", file=sys.stderr)
        return None

    try:
        return visualize_alignment(mix_id, pkl_path, full_viz_dir, full_meta_dir)
    except Exception as e:
        print(f"Error during visualization for mix {mix_id}: {e}", file=sys.stderr)
        return None