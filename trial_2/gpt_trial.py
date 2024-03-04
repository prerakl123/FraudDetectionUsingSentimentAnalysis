import cv2
from cv2.data import haarcascades
from deepface import DeepFace

# Initialize a dictionary to store information about each person
people_dict = {}

# Initialize a counter to keep track of the total number of people
total_people = 0

# Initialize the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the video
video_path = "video_analysis/video_1.mp4"
video_capture = cv2.VideoCapture(video_path)

# Initialize directory to store frames
output_directory = "video_analysis/output_frames/"
frame_count = 0

# Loop through each frame of the video
while video_capture.isOpened():
    # Read the next frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Increment frame counter
    frame_count += 1

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Increment total people count
        total_people += 1

        # Store frame with person's name
        person_name = f"person_{total_people}"
        frame_filename = f"{video_path.split('/')[-1]}_frame_{frame_count}.jpg"
        cv2.imwrite(output_directory + frame_filename, frame)

        # Update people dictionary
        if person_name not in people_dict:
            people_dict[person_name] = {
                "frame_counter": 1,
                "frame_list": [frame_count],
                "emotions": {}
            }
        else:
            people_dict[person_name]["frame_counter"] += 1
            people_dict[person_name]["frame_list"].append(frame_count)

    # # Display the frame with bounding boxes
    # cv2.imshow('Frame', frame)

    # # Exit if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

# Print the total number of people detected
print("Total People Detected:", total_people)

# Print the people dictionary
print("People Dictionary:")
print(people_dict)

# Loop through each person in the people dictionary
for person_name in people_dict:
    # Extract emotions from the person's face in each frame
    for frame_number in people_dict[person_name]["frame_list"]:
        # Read the frame
        frame_filename = f"{video_path.split('/')[-1]}_frame_{frame_number}.jpg"
        frame_path = output_directory + frame_filename
        frame = cv2.imread(frame_path)

        # Extract emotions using DeepFace
        emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Update the emotions in the people dictionary
        people_dict[person_name]["emotions"][frame_number] = emotions[0]
