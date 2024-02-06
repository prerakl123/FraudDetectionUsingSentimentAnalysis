import cv2
import cv2.data


def read_video(video_path):
    """Reads a video and returns a list of frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def resize_frames(frames, target_size=(224, 224)):
    """Resizes a list of frames to a consistent size."""
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, target_size)
        resized_frames.append(resized_frame)
    return resized_frames


def extract_faces(frames):
    """Extracts facial regions from frames using pre-trained Haarcascade face detector."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_frame = frame[y:y + h, x:x + w]
            face_frames.append(face_frame)

    return face_frames
