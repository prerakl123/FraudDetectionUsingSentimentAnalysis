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
