# import face_recognition
# import os
# import sys
# import cv2
# import numpy as np
# import math
#
#
# def face_confidence(face_distance, face_match_threshold=0.6):
#     _range = (1.0 - face_match_threshold)
#     linear_value = (1.0 / face_distance) / (_range * 2.0)
#     if face_distance > face_match_threshold:
#         return str(round(linear_value * 100, 2)) + '%'
#     else:
#         value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
#         return str(round(value, 2)) + '%'
#
#
# class FaceRecognition:
#     face_locations = []
#     face_encodings = []
#     face_names = []
#     known_face_encodings = []
#     known_face_names = []
#     process_current_frame = True
#
#     def __init__(self):
#         self.encode_faces()
#
#     def encode_faces(self):
#         for image in os.listdir('../images'):
#             face_image = face_recognition.load_image_file(f"../images/{image}")
#             face_encoding = face_recognition.face_encodings(face_image)[0]
#
#             self.known_face_encodings.append(face_encoding)
#             self.known_face_names.append(image)
#
#         print(self.known_face_names)
#
#     def run_recognition(self):
#         video_capture = cv2.VideoCapture('../videos/video_2.mp4')
#         if not video_capture.isOpened():
#             sys.exit('Video source not found')
#
#         while True:
#             ret, frame = video_capture.read()
#             if self.process_current_frame:
#                 small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#                 rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
#
#                 # Find all the faces in the current frame
#                 self.face_locations = face_recognition.face_locations(rgb_small_frame)
#                 self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
#
#                 self.face_names = []
#                 for face_encoding in self.face_encodings:
#                     matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#                     name = 'Unknown'
#                     confidence = 'Unknown'
#
#                     face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
#                     best_match_index = np.argmin(face_distances)
#
#                     if matches[best_match_index]:
#                         name = self.known_face_names[best_match_index]
#                         confidence = face_confidence(face_distances[best_match_index])
#
#                     self.face_names.append(f"{name} ({confidence})")
#
#             self.process_current_frame = not self.process_current_frame
#
#             # Display annotations
#             for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
#                 top *= 4
#                 right *= 4
#                 bottom *= 4
#                 left *= 4
#
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#                 cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
#                 cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
#
#             cv2.imshow('Face Recognition', frame)
#             if cv2.waitKey(1) == ord('q'):
#                 break
#
#         video_capture.release()
#         cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     fr = FaceRecognition()
#     fr.run_recognition()


# from deepface import DeepFace
# from pprint import pprint
#
# models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib"]
# backends = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8']
# cd = {}
#
# for m in models:
#     cd[m] = {}
#
#     for b in backends:
#         obj = DeepFace.verify("../images/elon.jpg", "../images/elon2.jpg", model_name=m, detector_backend=b)
#         cd[m][b] = (obj['threshold'], obj['time'])
#
# pprint(cd)
#
# DeepFace.stream(db_path="./db", source="../videos/video_2.mp4", time_threshold=1, frame_threshold=5,
#                 detector_backend='retinaface')


# cd2 = {'DeepFace': {'dlib': (0.23, 3.61),
#                     'mediapipe': (0.23, 2.51),
#                     'mtcnn': (0.23, 4.09),
#                     'opencv': (0.23, 10.96),
#                     'retinaface': (0.23, 4.8),
#                     'ssd': (0.23, 2.85),
#                     'yolov8': (0.23, 2.58)},
#
#        'DeepID': {'dlib': (0.015, 1.11),
#                   'mediapipe': (0.015, 0.05),
#                   'mtcnn': (0.015, 1.69),
#                   'opencv': (0.015, 1.07),
#                   'retinaface': (0.015, 2.04),
#                   'ssd': (0.015, 0.34),
#                   'yolov8': (0.015, 0.11)},
#
#        'Dlib': {'dlib': (0.07, 1.89),
#                 'mediapipe': (0.07, 0.8),
#                 'mtcnn': (0.07, 2.37),
#                 'opencv': (0.07, 1.77),
#                 'retinaface': (0.07, 2.86),
#                 'ssd': (0.07, 1.08),
#                 'yolov8': (0.07, 0.87)},
#
#        'Facenet': {'dlib': (0.4, 1.44),
#                    'mediapipe': (0.4, 0.38),
#                    'mtcnn': (0.4, 1.95),
#                    'opencv': (0.4, 3.75),
#                    'retinaface': (0.4, 2.43),
#                    'ssd': (0.4, 0.67),
#                    'yolov8': (0.4, 0.44)},
#
#        'OpenFace': {'dlib': (0.1, 1.23),
#                     'mediapipe': (0.1, 0.16),
#                     'mtcnn': (0.1, 1.74),
#                     'opencv': (0.1, 1.98),
#                     'retinaface': (0.1, 2.24),
#                     'ssd': (0.1, 0.46),
#                     'yolov8': (0.1, 0.22)},
#
#        'VGG-Face': {'dlib': (0.68, 1.83),
#                     'mediapipe': (0.68, 1.38),
#                     'mtcnn': (0.68, 2.59),
#                     'opencv': (0.68, 3.1),
#                     'retinaface': (0.68, 5.66),
#                     'ssd': (0.68, 0.65),
#                     'yolov8': (0.68, 10.76)}}
#
# for i in cd2.keys():
#     max_thres = -1
#     max_thres_name = ''
#     min_time = 10e7
#     min_time_name = ''
#
#     for j in cd2[i].keys():
#         if cd2[i][j][0] > max_thres:
#             max_thres = cd2[i][j][0]
#             max_thres_name = j
#
#         if cd2[i][j][1] < min_time:
#             min_time = cd2[i][j][1]
#             min_time_name = j
#
#     print("Max thres for", i, "model with backend", max_thres_name, ":", max_thres)
#     print("Min time for", i, "model with backend", min_time_name, ":", min_time)
#     print()


from deepface import DeepFace
import cv2
import json
import threading
import time

is_running = True
unique_faces = {}
unique_face_images = []
backend = 'retinaface'
total_frames = 0
frame_count = 0
VIDEO_DIR = "../videos"
VIDEO_FILE = 'video_7.mp4'
VIDEO_ANALYSIS_DIR = "./video_analysis"
JSON_PICS_DIR = 'video_analysis/json_pics_video_7'


def update_dictionary(filename=f'{VIDEO_ANALYSIS_DIR}/{VIDEO_FILE.split(".")[0]}_analysis.json'):
    global is_running, unique_faces

    while is_running:
        with open(filename, 'w') as f:
            json.dump(unique_faces, f, indent=4)
        f.close()
        print(total_frames, '|', frame_count)
        time.sleep(5)


def run():
    start_time = time.time()
    global unique_faces, unique_face_images, is_running, backend, total_frames, frame_count

    vc = cv2.VideoCapture(f"{VIDEO_DIR}/{VIDEO_FILE}")
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print("TOTAL FRAMES:", frame_count)

    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            is_running = False
            break

        total_frames += 1

        try:
            # print(total_frames)
            face_objs = DeepFace.extract_faces(img_path=frame, detector_backend=backend, enforce_detection=False)
            print(face_objs[0]['facial_area'])
            for face in face_objs:
                x = face['facial_area']['x']
                y = face['facial_area']['y']
                w = face['facial_area']['w']
                h = face['facial_area']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            cv2.imshow('frame', frame)
            if cv2.waitKey(5) == ord('q'):
                break

            if len(unique_faces) == 0:
                unique_faces = {
                    i: {
                        "count": 1,
                        "frames": []
                    } for i in range(len(face_objs))
                }
                unique_face_images = [(face['face']*255)[:, :, ::-1] for face in face_objs]

                for i, img_data in enumerate(unique_face_images):
                    cv2.imwrite(f'{JSON_PICS_DIR}/person - {i}.jpg', img_data)
            else:
                for i, face in enumerate(face_objs):
                    matched = False
                    face_rgb = face['face']
                    face_rgb *= 255
                    face_rgb = face_rgb[:, :, ::-1]

                    for img in unique_face_images:
                        try:
                            face_verify = DeepFace.verify(
                                face_rgb, img,
                                detector_backend=backend,
                                enforce_detection=False
                            )

                            if face_verify['verified']:
                                print(face_verify['verified'])
                                matched = True
                                unique_faces[i]['count'] += 1
                                emotions = DeepFace.analyze(
                                    face_rgb,
                                    actions=['emotion'],
                                    enforce_detection=True,
                                    detector_backend=backend
                                )
                                unique_faces[i]['frames'].append({
                                    'emotions': emotions[0],
                                    'frame_index': total_frames
                                })
                                break
                        except ValueError:
                            print(
                                "No face detected in frame:",
                                total_frames,
                                "for the unique face index:",
                                unique_face_images.index(img)
                            )
                            continue

                    if not matched:
                        new_ind = len(unique_faces)

                        unique_faces[new_ind] = {
                            'count': 1,
                            'frames': []
                        }
                        unique_face_images.append(face_rgb)

                        try:
                            emotions = DeepFace.analyze(
                                face_rgb,
                                actions=['emotion'],
                                enforce_detection=True,
                                detector_backend=backend
                            )
                            unique_faces[new_ind]['frames'].append({
                                'emotions': emotions[0],
                                'frame_index': total_frames
                            })
                            cv2.imwrite(f'{JSON_PICS_DIR}/person - {new_ind}.jpg', unique_face_images[new_ind])
                        except ValueError:
                            print("No faces matched and detected.")
                            continue
        except ValueError:
            print("No face detected in frame:", total_frames)
            continue

    is_running = False
    vc.release()
    cv2.destroyAllWindows()
    print("Total Time Elapsed: %.2f mins" % ((time.time() - start_time) / 60))


if __name__ == '__main__':
    s = [
        threading.Thread(target=run, name="p1"),
        threading.Thread(target=update_dictionary, name="p2")
    ]

    for animal in s:
        animal.start()
        print(animal.name)

    for animal in s:
        animal.join()
