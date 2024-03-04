import cv2
from cv2.data import haarcascades
import numpy as np


def detect_person_in_video(image_path, video_path):
    # Load the image
    person_image = cv2.imread(image_path)

    # Convert the image to grayscale
    person_gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

    # Load the video
    video_capture = cv2.VideoCapture(video_path)

    # Initialize a face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    total_frames = 0
    frames_with_person = 0

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        if not ret:
            break  # Break the loop if no frame is returned

        total_frames += 1

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if the person's image is present in the frame
        for (x, y, w, h) in faces:
            roi_gray = frame_gray[y:y+h, x:x+w]

            # Compare the person's image with the detected face region
            result = cv2.matchTemplate(roi_gray, person_gray, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8  # Adjust this threshold as needed

            if np.max(result) >= threshold:
                frames_with_person += 1
                # Optionally, you can draw a rectangle around the detected face
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Release the video capture object
    video_capture.release()

    # Calculate the percentage of frames with the person
    percentage_frames_with_person = (frames_with_person / total_frames) * 100

    print(f"Person appeared in {frames_with_person} out of {total_frames} frames.")
    print(f"Percentage of frames with the person: {percentage_frames_with_person:.2f}%")


# Example usage:
detect_person_in_video("person_image.jpg", "video.mp4")
