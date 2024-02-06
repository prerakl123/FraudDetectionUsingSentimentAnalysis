import os

import whisper
from moviepy.audio.fx.volumex import volumex
from moviepy.editor import VideoFileClip, AudioFileClip
from constants import VIDEO_DIR, AUDIO_DIR, WHISPER_MODEL

whisper_model = whisper.load_model(WHISPER_MODEL)


def list_videos():
    """List all videos in the `VIDEO_DIR`"""
    return list(map(
        lambda vn: f"{VIDEO_DIR}/{vn}",
        os.listdir(VIDEO_DIR)
    ))


def extract_audio(video_file_name: str):
    video = VideoFileClip(filename=video_file_name)
    audio = video.audio
    audio_file = f"{AUDIO_DIR}/{video.filename.split('/')[-1].split('.')[0]}.mp3"
    audio.write_audiofile(audio_file)
    return audio_file


def multiply_volume(audio_file_name: str):
    afc = AudioFileClip(audio_file_name)
    afc.fx(volumex, 2.0)
    afc.write_audiofile(audio_file_name)


def extract_text(audio_file_name: str):
    audio = whisper.load_audio(audio_file_name)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions(language='en', fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    return result.text


if __name__ == '__main__':
    vids = list_videos()
    print(vids)
    af_name = extract_audio(f"{vids[0]}")
    print(af_name)
    print(extract_text(af_name))
