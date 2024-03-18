import random
import time
from functools import cache, lru_cache
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from cv2.data import haarcascades
from deepface import DeepFace
from numpy import ndarray

hcc = cv2.CascadeClassifier(haarcascades + 'haarcascade_frontalface_default.xml')


class EmotionParser:
    edict: dict
    unique_faces: list

    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.file_name = self.video_path.name
        self.video_capture = cv2.VideoCapture(video_path)
        self.video_dimen = [
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ]
        print("Total Frames:", int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.running = True
        self.current_frame = 0

    def find_unique(self):
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break

            self.current_frame += 1
            print(self.current_frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coords = hcc.detectMultiScale(gray, 1.2, 6, minSize=(100, 100))
            detected_faces = []
            black_screen = np.zeros((self.video_dimen[1], self.video_dimen[0], 3), dtype=np.uint8) * 255

            for (x, y, w, h) in face_coords:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                black_screen[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
                detected_faces.append(frame[y:y+h, x:x+w])

            if face_coords.__len__() > 1:
                a = random.randint(0, 1)
                b = random.randint(0, 1)
                st = time.time()
                print(self.compare(
                    detected_faces[a], detected_faces[b]
                ))
                print(f"{a} == {b} == {a == b:}")
                print("Time to verify:", time.time() - st)

            cv2.imshow("Faces", black_screen)
            cv2.imshow("Original", frame)

            if cv2.waitKey(15) == ord('q'):
                break

    def compare(self, img1, img2):
        result = DeepFace.verify(img1, img2)
        return result

    def save_img(self, *img_arrays: ndarray):
        for i, img in enumerate(img_arrays):
            cv2.imwrite(self.file_name.split('.')[0] + f" - {i}.jpg")

    def update_edict(self):
        pass

    def run(self):
        while self.running:
            pass


if __name__ == '__main__':
    e = EmotionParser("../videos/video_6.mp4")
    e.find_unique()
