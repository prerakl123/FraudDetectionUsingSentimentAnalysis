# PyAnnote Diarization
import os.path
import time

from moviepy.audio.io.AudioFileClip import AudioFileClip
from pyannote.audio import Pipeline
import torch

# Save Audio File from Video
video_file_name = 'video_1'
audio_file_path = f'./audio_clips/{video_file_name}_mp4.mp3'
if not os.path.isfile(f'./audio_clips/{video_file_name}_mp4.mp3'):
    clip = AudioFileClip(f'../videos/{video_file_name}.mp4')
    clip.write_audiofile(audio_file_path)
    print("Audio file duration:", clip.duration)
    clip.close()
else:
    clip = AudioFileClip(audio_file_path)
    print("Audio file duration:", clip.duration)
    clip.close()

# Diarization
HGF_TOKEN = "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"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HGF_TOKEN)

# send pipeline to GPU
pipeline.to(torch.device("cuda"))

start_time = time.time()
# apply pretrained pipeline
diarization = pipeline(audio_file_path)
print("Diarization Elapsed Time:", time.time() - start_time)

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s")
