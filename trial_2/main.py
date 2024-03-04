import cv2
from cv2.data import haarcascades


video_file = "video_analysis/video_1.mp4"
output_dir = "video_analysis/output_frames"

haarcascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_cap = cv2.VideoCapture(video_file)
frame_count = 0

while video_cap.isOpened():
    ret, frame = video_cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haarcascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
