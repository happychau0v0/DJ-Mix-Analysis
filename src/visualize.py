import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse
import numpy as np
import librosa  # Added for beat tracking

# Assuming COLORS and MATCH_RATE_THRESHOLD are defined elsewhere
COLORS = ['red', 'blue', 'green', 'purple', 'orange']  # Example colors
MATCH_RATE_THRESHOLD = 0.8  # Example threshold

# New function to get mix beat positions
def get_mix_beats(mix_id):
    """
    Compute the beat positions of the mix audio file in seconds.
    
    Args:
        mix_id: ID of the mix
        
    Returns:
        Array of beat times in seconds, or None if the mix WAV file is not found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mix_wav_path = os.path.join(script_dir, '..', 'data', 'wav', mix_id, 'mix.wav')
    if not os.path.exists(mix_wav_path):
        print(f"Mix WAV file not found: {mix_wav_path}")
        return None
    y, sr = librosa.load(mix_wav_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    return beats

# New function to convert timestamps to seconds (replacing inline logic)
def timestamp_to_seconds(timestamp):
    """
    Convert a timestamp string to seconds.
    
    Args:
        timestamp: String in 'H:MM:SS' or 'MM:SS' format
        
    Returns:
        Total seconds as float
        
    Raises:
        ValueError: If the timestamp format is invalid
    """
    time_parts = timestamp.split(':')
    if len(time_parts) == 3:  # H:MM:SS
        hours, minutes, seconds = map(int, time_parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(time_parts) == 2:  # MM:SS
        minutes, seconds = map(int, time_parts)
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

def visualize_alignment(mix_id, pkl_path, viz_dir):
    """
    Create visualization of alignment paths styled like the worker function, including ground truth timestamps.
    
    Args:
        mix_id: ID of the mix
        pkl_path: Path to the alignment .pkl file
        viz_dir: Directory to save visualizations
    """
    try:
        results_df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"Error reading .pkl file at {pkl_path}: {e}")
        return
    if results_df.empty:
        print(f"No results to visualize for mix {mix_id}")
        return
    
    # Read ground truth CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(script_dir, '..', 'data', 'meta', f"{mix_id}.csv")
    try:
        gt_df = pd.read_csv(gt_path)
    except Exception as e:
        print(f"Error reading ground truth CSV at {gt_path}: {e}")
        gt_df = pd.DataFrame()  # Empty DataFrame if CSV fails
    
    # Get max values for plot dimensions
    xmax = max(results_df['mix_cue_out_beat'].max(), results_df['mix_cue_in_beat'].max())
    ymax = max(results_df['track_cue_out_beat'].max(), results_df['track_cue_in_beat'].max())
    
    plt.figure(figsize=(int(xmax / ymax * 2), 4))  # Match worker's figure size
    
    for i, (_, track) in enumerate(results_df.iterrows()):
        color = COLORS[i % len(COLORS)]
        wp = track['wp']
        
        # Plot the warping path
        plt.plot(wp[:, 1], wp[:, 0], color=color)
        
        # Add track label at the start of the warping path
        textoffset = 5
        plt.text(
            wp[0, 1] + textoffset, 
            wp[0, 0] + textoffset, 
            f"{track['i_track']}", 
            color='white',
            bbox={'facecolor': color, 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
            verticalalignment='bottom', 
            horizontalalignment='left'
        )
        
        # Plot cue points as vertical lines
        plt.axvline(track['mix_cue_in_beat'], color=color, linestyle='--', linewidth=1)
        plt.axvline(track['mix_cue_out_beat'], color=color, linestyle='--', linewidth=1)
        
        # Add text labels for cue points
        plt.text(
            track['mix_cue_in_beat'] + textoffset, 
            ymax * 1.10 + textoffset, 
            f"{track['i_track']}_in", 
            color='white',
            bbox={'facecolor': color, 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
            verticalalignment='bottom', 
            horizontalalignment='left'
        )
        plt.text(
            track['mix_cue_out_beat'] + textoffset, 
            ymax * 1.10 + textoffset, 
            f"{track['i_track']}_out", 
            color='white',
            bbox={'facecolor': color, 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
            verticalalignment='bottom', 
            horizontalalignment='left'
        )
        
        # Draw match quality indicator at the bottom
        if track['match_rate'] > MATCH_RATE_THRESHOLD:
            box_height = 0.04 * ymax
            plt.gca().add_patch(mpatches.Rectangle(
                xy=(track['mix_cue_in_beat'], box_height * i),
                width=track['mix_cue_out_beat'] - track['mix_cue_in_beat'], 
                height=box_height,
                linewidth=0, 
                facecolor=color, 
                alpha=0.5
            ))
    
    # Get mix beats for mapping timestamps to beat frames
    mix_beats = get_mix_beats(mix_id)
    
    # Plot ground truth timestamps as black vertical lines, mapped to beat frames
    if not gt_df.empty:
        if mix_beats is None:
            print("Cannot plot ground truth timestamps without mix beats.")
        else:
            for _, gt_row in gt_df.iterrows():
                i_track = gt_row['i_track']
                timestamp = gt_row['timestamp']
                if pd.notna(timestamp):
                    try:
                        timestamp_secs = timestamp_to_seconds(timestamp)
                        # Find the closest beat index
                        i_beat = np.argmin(np.abs(mix_beats - timestamp_secs))
                        # Plot ground truth line at the beat index
                        plt.axvline(i_beat, color='black', linestyle='-', linewidth=1.5)
                        plt.text(
                            i_beat + textoffset, 
                            ymax * 1.05,  # Slightly lower than cue labels
                            f"GT_{i_track}", 
                            color='white',
                            bbox={'facecolor': 'black', 'boxstyle': 'square, pad=0.1', 'linewidth': 0},
                            verticalalignment='bottom', 
                            horizontalalignment='left'
                        )
                    except ValueError as e:
                        print(f"Invalid timestamp format for track {i_track}: {timestamp} - {e}")
    
    # Set plot limits
    plt.xlim(0, xmax * 1.03)
    plt.ylim(0, ymax * 1.17)
    
    # Create legend with track info
    legend_handles = []
    
    # Add ground truth patch to legend
    legend_handles.append(mpatches.Patch(
        color='black', 
        label='Ground Truth'
    ))
    
    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(0., 1.02, 1., .102), 
        loc='lower left', 
        ncol=3, 
        mode='expand', 
        borderaxespad=0.
    )
    
    # Add labels
    plt.ylabel('Track beat frame')
    plt.xlabel('Mix beat frame')
    
    plt.tight_layout()
    viz_path = os.path.join(viz_dir, f"{mix_id}.pdf")
    plt.savefig(viz_path)
    print(f"Saved visualization to {viz_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize DTW alignment paths from a .pkl file with ground truth')
    parser.add_argument('mix_id', help='ID of the mix to visualize (e.g., 6qdzkf9)')
    parser.add_argument('--pkl-path', default=None, help='Path to the alignment .pkl file (default: ../data/align/[mix_id]-chroma+mfcc-keyinv.pkl)')
    parser.add_argument('--viz-dir', default='../data/dtwviz', help='Directory to save the visualization (default: ../data/dtwviz)')
    args = parser.parse_args()

    # Construct default .pkl path if not provided
    if args.pkl_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(script_dir, '..', 'data', 'align', f"{args.mix_id}-chroma+mfcc.pkl")
    else:
        pkl_path = args.pkl_path

    # Check if the .pkl file exists
    if not os.path.exists(pkl_path):
        print(f"Alignment .pkl file not found at {pkl_path}")
        return

    # Ensure viz_dir exists
    os.makedirs(args.viz_dir, exist_ok=True)
    
    # Visualize the alignment
    visualize_alignment(args.mix_id, pkl_path, args.viz_dir)

if __name__ == '__main__':
    main()