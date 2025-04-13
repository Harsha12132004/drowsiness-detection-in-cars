import cv2
import mediapipe as mp
import pyttsx3
from scipy.spatial import distance
import time
import sys
sys.path.append(r"C:\Users\ASUS\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages")

# INITIALIZING pyttsx3 FOR AUDIO ALERT
engine = pyttsx3.init()

# INITIALIZING MEDIAPIPE FACE MESH
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # Use refine_landmarks to improve accuracy
mp_drawing = mp.solutions.drawing_utils

# FUNCTION TO CALCULATE EYE ASPECT RATIO (EAR)
def detect_eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_eye

# CAMERA SETUP
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# VARIABLES TO TRACK DROWSINESS
CLOSED_EYES_FRAME_COUNT = 0
EYE_CLOSED_TIME_THRESHOLD = 2  # seconds
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback to 30 if FPS not available
closed_eye_frame_threshold = int(fps * EYE_CLOSED_TIME_THRESHOLD)  # Number of frames for 2 seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Convert the frame to RGB as Mediapipe requires
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Extract landmarks for left and right eyes
            left_eye_landmarks = [
                (face_landmarks.landmark[i].x * frame.shape[1], 
                 face_landmarks.landmark[i].y * frame.shape[0]) 
                for i in [362, 385, 387, 263, 373, 380]  # Right Eye Mediapipe Indices
            ]
            right_eye_landmarks = [
                (face_landmarks.landmark[i].x * frame.shape[1], 
                 face_landmarks.landmark[i].y * frame.shape[0]) 
                for i in [33, 160, 158, 133, 153, 144]  # Left Eye Mediapipe Indices
            ]

            # Calculate Eye Aspect Ratio (EAR)
            right_eye = detect_eye(right_eye_landmarks)
            left_eye = detect_eye(left_eye_landmarks)
            eye_rat = (left_eye + right_eye) / 2

            # Display EAR value on screen
            cv2.putText(frame, f"EAR: {eye_rat:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2)

            # Check for drowsiness
            if eye_rat < 0.25:  # Threshold for closed eyes
                CLOSED_EYES_FRAME_COUNT += 1
                if CLOSED_EYES_FRAME_COUNT >= closed_eye_frame_threshold:
                    # Alert for drowsiness
                    cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Audio Alert
                    engine.say("Alert!!!! WAKE UP DUDE")
                    engine.runAndWait()
            else:
                CLOSED_EYES_FRAME_COUNT = 0  # Reset counter if eyes are open

    # Show the frame
    cv2.imshow("Drowsiness Detection with Mediapipe", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()