import pandas as pd
import numpy as np
import librosa

SR = 22050

def beats(path):
    y, sr = librosa.load(path)
    tempo, beats_ = librosa.beat.beat_track(y=y, sr=sr, units='time')
    return beats_


def mfcc(path):
  sig_, sr = librosa.load(path)
  mfcc_ = librosa.feature.mfcc(sig_, sr, n_mfcc=12)
  return mfcc_


def beat_mfcc(path):
  beats_ = beats(path)
  mfcc_ = mfcc(path)
  beat_mfcc_ = beat_aggregate(mfcc_, beats_)
  return beat_mfcc_


def chroma_cens(path):
  sig, sr = librosa.load(path)
  chroma_cens_ = librosa.feature.chroma_cens(sig, sr)
  return chroma_cens_


def beat_chroma_cens(path):
  beats_ = beats(path)
  chroma_cens_ = chroma_cens(path)
  beat_chroma_cens_ = beat_aggregate(chroma_cens_, beats_)
  return beat_chroma_cens_


def beat_aggregate(feature, beats, frames_per_beat=None):
  max_frame = feature.shape[1]
  beat_frames = librosa.time_to_frames(beats)
  beat_frames = beat_frames[beat_frames < max_frame]
  beat_feature = np.split(feature, beat_frames, axis=1)
  # Average for each beat.
  beat_feature = beat_feature[1:-1]  # only use chroma features between beats. not before or after beat
  beat_feature = [f.mean(axis=1) for f in beat_feature]  # average chroma features for each beat
  beat_feature = np.array(beat_feature).T
  return beat_feature


def main():
  with Pool() as pool:
    paths = [f'data/mix/{mix_id}.wav' for mix_id in df_tlist.mix_id.unique()]
    paths += ['data/track/' + filename for filename in df_tlist.filename]
    iterator = pool.imap(extract_feature, paths)
    for _ in tqdm(iterator, total=len(paths)):
      pass


def extract_feature(path):
  beat_chroma_cens(path)
  beat_mfcc(path)


if __name__ == '__main__':
  main()
