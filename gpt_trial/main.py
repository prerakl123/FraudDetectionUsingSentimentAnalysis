import os
import sys
from pathlib import Path

import torch
import whisper
from moviepy.audio.io.AudioFileClip import AudioFileClip
from pyannote.audio import Pipeline

HGF_TOKEN = "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"
PRETRAINED_MODEL_NAME = "pyannote/speaker-diarization-3.1"
audio_clips_path = Path('./audio_clips').resolve()

WHISPER_MODEL = 'base'


def make_dir(dirname: Path = None):
    dirname = dirname.resolve()

    if not os.path.isdir(audio_clips_path):
        print("`audio_clips` DNE. Creating `audio_clips`...", file=sys.stderr, end=' ')
        os.makedirs(audio_clips_path, exist_ok=True)
        print('Done.')

    if dirname:
        if not os.path.isdir(dirname):
            print(f"Creating {dirname.name}...", file=sys.stderr, end=' ')
            os.makedirs(dirname, exist_ok=True)
            print('Done.')


def get_audio_meta(video_file_path: Path) -> dict:
    audio_clips_dir = audio_clips_path / video_file_path.name.replace('.', '_')
    audio_clip_name = audio_clips_dir / "audio.wav"
    make_dir(audio_clips_dir)

    print('Extracting Audio from:', video_file_path.as_posix())
    clip = AudioFileClip(video_file_path.as_posix())
    clip.write_audiofile(audio_clip_name.as_posix())
    dur = clip.duration
    clip.close()
    print("Saved audio to:", audio_clip_name.as_posix())

    return {
        "path": audio_clip_name,
        "length": dur
    }

