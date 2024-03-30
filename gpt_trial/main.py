import os
import sys
from pathlib import Path

import google.generativeai as genai
import openai
import torch
import whisper
from moviepy.audio.io.AudioFileClip import AudioFileClip
from pyannote.audio import Pipeline

import time
from pprint import pprint

HGF_TOKEN = "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"
PRETRAINED_MODEL_NAME = "pyannote/speaker-diarization-3.1"
audio_clips_path = Path('./audio_clips').resolve()

WHISPER_MODEL = 'large'

GEMINI_API_KEY = 'AIzaSyBhgicqSBzKJZPIQA2w-ahp_7YACYCNBVM'
OPENAI_API_KEY = ''

CONV_GEN_CONTEXT = open('conv_gen_context.txt', 'r').read()
SCORE_GEN_CONTEXT = open('score_gen_context.txt', 'r').read()
TRANSCRIPT_FORMAT = '{start_time}->{end_time} = "{text}"'

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH"
    }
]
GEMINI_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1
}
genai.configure(api_key=GEMINI_API_KEY)
MODEL_GEMINI = genai.GenerativeModel(
    model_name='gemini-1.0-pro',
    generation_config=genai.GenerationConfig(**GEMINI_GENERATION_CONFIG),
    safety_settings=SAFETY_SETTINGS
)


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
            print('Done.', file=sys.stderr)


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
        "durations": [
            (turn.start, turn.end) for turn, _ in diar.itertracks(yield_label=False)
        ]
    }


def get_subclips(audio_file_path: Path, durations: list) -> list:
    audio_dir = audio_file_path.parent
    subclips = []

    clip = AudioFileClip(audio_file_path.as_posix())
    for i, time in enumerate(durations):
        subclip_name = audio_dir / f'subclip_{i}.wav'
        clip.subclip(time[0], time[1]).write_audiofile(subclip_name)
        subclips.append((time[0], time[1], subclip_name.absolute()))
    clip.close()

    return subclips


def get_transcripts(
        audio_file_path: Path = None,
        video_file_path: Path = None,
        durations: list[tuple[float, float]] = None,
        whisper_model=None,
        cuda=False
):
    if durations is None:
        print("Durations not provided. Getting durations...", file=sys.stderr)

        if video_file_path is not None:
            audio_file_path, length, durations = get_durations(video_file_path=video_file_path, cuda=cuda).values()

    if whisper_model is None:
        print("Using default whisper model:", WHISPER_MODEL, file=sys.stderr)
        whisper_model = WHISPER_MODEL

    if cuda:
        print("Setting CUDA=1.", file=sys.stderr, end=' ')
        model = whisper.load_model(whisper_model, device='cuda')
        print("Using device:", torch.cuda.get_device_name(), file=sys.stderr)
    else:
        model = whisper.load_model(whisper_model, device='cpu')
        print('Whisper model running on CPU.', file=sys.stderr)

    print("Creating Subclips...")
    subclips = get_subclips(audio_file_path, durations)

    transcriptions: list = []

    print("Transcribing text...", end=' ')
    for start, end, subclip_name in subclips:
        print(f"Transcribing...: {subclip_name.as_posix()}.", file=sys.stderr, end=' ')
        transcript = model.transcribe(subclip_name.as_posix())
        transcriptions.append((start, end, transcript['text']))
        print("Done.", file=sys.stderr)

    return transcriptions


def get_convo_prompt(transcripts, **kwargs):
    print("Generating prompt...", end=" ")

    transcript_list = []
    extra_kws = ''
    kw_str_list = []

    if kwargs:
        kw_str_list = [f"{key}: {val}" for key, val in kwargs.items()]
        extra_kws = '\n'.join(kw_str_list)

    for start, end, text in transcripts:
        transcript_list.append(
            TRANSCRIPT_FORMAT.format(start_time=start, end_time=end, text=text)
        )

    audio_transcript_data = '\n'.join(transcript_list)

    print("DONE. ({lines} lines)".format(
        lines=len(transcript_list) + len(kw_str_list)
    ))
    return CONV_GEN_CONTEXT.format(audio_transcript_data=audio_transcript_data, extra_kws=extra_kws)


def get_response(prompt: str = None, history: list = None, return_history=True, gpt=False, **gpt_config):
    if prompt == history is None:
        raise RuntimeError('Either prompt or history should be provided!')

    if history is None:
        history = []

    if prompt is not None:
        if not gpt:
            prompt_dict = {
                "role": "user",
                "parts": [prompt]
            }
        else:
            prompt_dict = {
                "role": "user",
                "content": prompt
            }

        history.append(prompt_dict)

    if gpt:
        print("Getting GPT response...", end=' ')
        response = openai.chat.completions.create(messages=history, **gpt_config).choices[0].message.content
        history.append({
            "role": "assistant",
            "content": response
        })
    else:
        print("Getting Gemini response...", end=' ')
        response = MODEL_GEMINI.generate_content(history).text
        history.append({
            "role": "model",
            "parts": [response]
        })

    print("DONE. (history={})".format(len(history)))

    if return_history:
        return response, history
    return response


def analyze_video(video_path: str, save_results=True):
    path = Path(video_path).resolve()
    audio_path, length, durations = get_durations(path, cuda=True).values()
    transcripts = get_transcripts(audio_file_path=audio_path, durations=durations, cuda=True)
    convo_prompt = get_convo_prompt(transcripts, total_audio_duration=f"{length} sec")
    convo, history = get_response(convo_prompt)
    score_prompt = SCORE_GEN_CONTEXT
    score = get_response(score_prompt, history=history, return_history=False)

    if save_results:
        with open(path.name.replace('.', '_') + '.txt', 'w', encoding='utf-8') as result_file:
            try:
                result_file.write("CONVERSATION:\n\n")
                result_file.write(convo)
                result_file.write('\n\n------------------------------\n\n')
                result_file.write("ANALYSIS:\n\n")
                result_file.write(score)
                result_file.close()
            except UnicodeDecodeError as e:
                print(convo)
                print(score)
                print(e.reason, 'from:', e.start, '-', e.end)


def main():
    # # from pprint import pprint
    #
    # path = Path('../videos/video_7.mp4').resolve()
    # tr = get_transcripts(video_file_path=path, whisper_model='medium', cuda=True)
    # # pprint(tr, width=150, depth=2, indent=2)
    # print(make_prompt(tr))

    # analyze_video('../videos/video_5.mp4')

    ana_times = {}

    videos = ['../videos/video_7.mp4', '../videos/video_6.mp4', '../videos/video_5.mp4']
    for video in videos:
        start_time = time.time()
        analyze_video(video)
        ana_times[video] = time.time() - start_time

    pprint(ana_times, width=150, depth=4, indent=4)


if __name__ == '__main__':
    main()
