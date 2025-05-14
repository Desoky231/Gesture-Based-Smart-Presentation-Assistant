import cv2
import numpy as np
from deepface import DeepFace
import os
import time

# Configuration
EMBEDDING_DIR = "user_embeddings/"
FACE_DETECTION_SCALE = 1.1
FACE_DETECTION_NEIGHBORS = 5
MIN_FACE_SIZE = (100, 100)
COSINE_SIMILARITY_THRESHOLD = 0.7  # Higher is more strict
MIN_CONSECUTIVE_MATCHES = 3  # Require multiple consecutive matches
MODEL_NAME = "Facenet"

# Create embedding directory if not exists
if not os.path.exists(EMBEDDING_DIR):
    os.makedirs(EMBEDDING_DIR)

# Load OpenCV Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def register_user():
    cap = cv2.VideoCapture(0)
    user_name = input("Enter your name for registration: ").strip()
    
    print("Please look at the camera to capture your face...")
    embeddings = []
    frame_count = 0
    required_frames = 5  # Capture multiple frames for better registration

    while frame_count < required_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=FACE_DETECTION_SCALE, 
            minNeighbors=FACE_DETECTION_NEIGHBORS,
            minSize=MIN_FACE_SIZE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = frame[y:y + h, x:x + w]

            try:
                # Get embedding with alignment
                embedding = DeepFace.represent(
                    img_path=face_roi,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                    align=True
                )[0]["embedding"]
                embeddings.append(embedding)
                frame_count += 1
                print(f"Captured frame {frame_count}/{required_frames}")
                
                # Show feedback to user
                cv2.putText(frame, f"Capturing {frame_count}/{required_frames}", 
                           (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
            except Exception as e:
                print(f"Error in face recognition: {e}")

        cv2.imshow("Register User", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if embeddings:
        # Average multiple embeddings for more robust registration
        avg_embedding = np.mean(embeddings, axis=0)
        np.savez_compressed(os.path.join(EMBEDDING_DIR, f"{user_name}.npz"), embedding=avg_embedding)
        print(f"User {user_name} registered successfully with {len(embeddings)} samples!")
    else:
        print("Failed to capture any valid face images.")
