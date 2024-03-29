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


def get_durations(
        video_file_path: Path = None,
        audio_meta=None,
        hgf_token=None,
        hgf_model=None,
        cuda=False
):
    if audio_meta is None:
        audio_meta = get_audio_meta(video_file_path)

    if hgf_token is None:
        hgf_token = HGF_TOKEN

    if hgf_model is None:
        hgf_model = PRETRAINED_MODEL_NAME

    print("Initializing model:", hgf_model)
    pipeline = Pipeline.from_pretrained(hgf_model, use_auth_token=hgf_token)

    if cuda:
        print("Setting CUDA=1.", file=sys.stderr, end=' ')
        pipeline.to(torch.device('cuda'))
        print("Using device:", torch.cuda.get_device_name(), file=sys.stderr)

    print("Running Diarization...", end=' ')
    diar = pipeline(audio_meta['path'].as_posix())
    print('Done.')

    return {
        **audio_meta,
        # "path": audio_meta['path'],
        # "length": audio_meta['length'],
        "durations": [
            (turn.start, turn.end) for turn, _ in diar.itertracks(yield_label=False)
        ]
    }


def get_subclips(audio_file_path: Path, durations: list) -> list:
    audio_dir = audio_file_path.absolute().parent
    subclips = []

    clip = AudioFileClip(audio_file_path)
    for i, time in enumerate(durations):
        subclip_name = audio_dir / f'subclip_{i}.wav'
        clip.subclip(time[0], time[1]).write_audiofile(subclip_name)
        subclips.append((time[0], time[1], subclip_name))

    return subclips

