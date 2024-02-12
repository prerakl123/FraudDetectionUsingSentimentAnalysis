import json

import whisper
from moviepy.editor import *

whisper_model = whisper.load_model("base", device='cpu')


def split(video_path):
    audio_clip = AudioFileClip(video_path)
    n = round(audio_clip.duration)
    audio_clip.close()

    counter, start = 0, 0
    index = 60  # time interval

    flag_to_exit = False
    try:
        while True:
            audio_clip = AudioFileClip(video_path)

            # last loop
            if index >= n:
                flag_to_exit = True
                index = n

            temp = audio_clip.subclip(start, index)
            temp_saving_location = f"audio/video_1_mp4_chunked/video_1_mp4_{counter}.mp3"
            temp.write_audiofile(
                filename=temp_saving_location,
                verbose=False,
                logger=None
            )
            temp.close()
            audio_clip.close()

            counter += 1
            start = index
            if flag_to_exit is True:
                break

            index += 60
    except OSError as error_msg:
        print(f"Audio split completed with {error_msg=}")


def save_subtitles(list_of_subtitles):
    path_to_save = "audio/video_1_mp4_chunked/subtitles.txt"
    with open(path_to_save, 'a+') as f:
        for i in list_of_subtitles:
            f.write(f"{i}\n")


def stt(base_path_to_saved_files):
    list_of_files = os.listdir(base_path_to_saved_files)
    start, end = 0, 0

    id_counter = 0
    final_list_of_text = []
    for index in range(len(list_of_files) - 1):
        path_to_saved_file = os.path.join(base_path_to_saved_files, f"video_1_mp4_{index}.mp3")
        audio_clip = AudioFileClip(path_to_saved_file)
        duration = audio_clip.duration
        audio_clip.close()
        out = whisper_model.transcribe(path_to_saved_file)
        list_of_text = out['segments']
        for line in list_of_text:
            line['start'] += start
            line['end'] += start

            line['id'] = id_counter
            id_counter += 1

            final_list_of_text.append(line)

        start += duration

    for index in range(len(final_list_of_text)):
        data = final_list_of_text[index]
        if index + 1 >= len(final_list_of_text):
            break

        future_data = final_list_of_text[index + 1]
        data['end'] = future_data['start']
        data['duration'] = data['end'] - data['start']

    return final_list_of_text


def get_transcript_from_chunks(subtitle_file):
    text_list = []

    with open(subtitle_file, 'r') as sf:
        subtitles = sf.readlines()
        sf.close()

    for line in subtitles:
        text_list.append(
            line.split("'text': ")[1].split(", 'tokens'")[0].lstrip("'").lstrip('"').rstrip('"').rstrip("'")
        )

    return "".join(text_list)


if __name__ == '__main__':
    # import time
    # start_time = time.time()
    # split('videos/video_1.mp4')
    # save_subtitles(stt("audio/video_1_mp4_chunked/"))
    # print("Elapsed Time: ", time.time() - start_time, 's')
    print(get_transcript_from_chunks("audio/video_1_mp4_chunked/subtitles.txt"))
