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

def load_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDING_DIR):
        if file.endswith(".npz"):
            user_name = file.split(".")[0]
            embeddings[user_name] = np.load(os.path.join(EMBEDDING_DIR, file))["embedding"]
    return embeddings

def authenticate_user(face_img, saved_embeddings):
    try:
        live_embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            enforce_detection=False,
            align=True
        )[0]["embedding"]

        max_similarity = -1
        recognized_user = "Unknown"
        
        for user_name, saved_embedding in saved_embeddings.items():
            # Calculate cosine similarity (higher is better)
            similarity = np.dot(live_embedding, saved_embedding) / (
                np.linalg.norm(live_embedding) * np.linalg.norm(saved_embedding))
            
            if similarity > max_similarity:
                max_similarity = similarity
                recognized_user = user_name

        if max_similarity > COSINE_SIMILARITY_THRESHOLD:
            return recognized_user, max_similarity
        return "Unknown", max_similarity

    except Exception as e:
        print(f"Authentication Error: {e}")
        return "Unknown", 0

def authentication_system():
    cap = cv2.VideoCapture(0)
    saved_embeddings = load_embeddings()
    
    if not saved_embeddings:
        print("No registered users found. Please register users first.")
        return

    consecutive_matches = 0
    last_recognized_user = None
    auth_start_time = None
    authenticated_user = None

    while True:
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

            recognized_user, similarity = authenticate_user(face_roi, saved_embeddings)
            
            # Display similarity score for debugging
            cv2.putText(frame, f"Similarity: {similarity:.2f}", 
                       (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Consecutive match logic
            if recognized_user != "Unknown":
                if recognized_user == last_recognized_user:
                    consecutive_matches += 1
                else:
                    consecutive_matches = 1
                    auth_start_time = time.time()
                
                last_recognized_user = recognized_user
                
                # Check if we have enough consecutive matches
                if consecutive_matches >= MIN_CONSECUTIVE_MATCHES:
                    authenticated_user = recognized_user
                    auth_start_time = time.time()
            else:
                consecutive_matches = 0
                last_recognized_user = None
                if authenticated_user and time.time() - auth_start_time > 5:  # 5 seconds timeout
                    authenticated_user = None

            # Display authentication status
            if authenticated_user:
                cv2.putText(frame, f"Access Granted: {authenticated_user}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Access Denied", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("User Authentication System", frame)
        if authenticated_user:
            cap.release()
            cv2.destroyAllWindows()
            return True  # ✅ Success
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
