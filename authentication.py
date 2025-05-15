import os, time, cv2, numpy as np
from deepface import DeepFace
from keras_facenet import FaceNet          # <— add this import once

# -------------------- configuration --------------------
EMBEDDING_DIR              = "user_embeddings"
MODEL_NAME                 = "Facenet512"     # 512-D, same as enrolment script
EMBEDDING_SIZE             = 512              # sanity-check
FACE_DETECTION_SCALE       = 1.1
FACE_DETECTION_NEIGHBORS   = 5
MIN_FACE_SIZE              = (100, 100)
COSINE_SIMILARITY_THRESHOLD = 0.99
MIN_CONSECUTIVE_MATCHES     = 3
TIMEOUT_SECONDS             = 5
# -------------------------------------------------------

# preload model once – huge speed-up
MODEL = DeepFace.build_model(MODEL_NAME)

# OpenCV Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def load_embeddings(directory: str = EMBEDDING_DIR):
    """Load all .npz files that have the expected embedding length."""
    db = {}
    for file in os.listdir(directory):
        if not file.endswith(".npz"):
            continue
        user   = file.rsplit(".", 1)[0]
        vector = np.load(os.path.join(directory, file))["embedding"]
        if vector.size != EMBEDDING_SIZE:
            print(f"[WARN] {user} has {vector.size}-D vector – skipping")
            continue
        db[user] = vector
    return db
EMBEDDER = FaceNet()                       # load once, like DeepFace model

def authenticate_user(face_img: np.ndarray, saved_db: dict):
    """Return (user, similarity) or ('Unknown', score) using the SAME
    keras-facenet model that created the stored templates."""
    try:
        # convert BGR → RGB, resize to 160×160 to match FaceNet input
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (160, 160)).astype(np.float32) / 255.0

        live_vec = EMBEDDER.embeddings([rgb])[0]           # 512-D
        live_vec /= np.linalg.norm(live_vec)               # L2-normalise

        best_user, best_sim = "Unknown", -1
        for user, saved_vec in saved_db.items():
            sim = _cosine(live_vec, saved_vec)
            if sim > best_sim:
                best_user, best_sim = user, sim

        return (best_user, best_sim) if best_sim > COSINE_SIMILARITY_THRESHOLD \
               else ("Unknown", best_sim)

    except Exception as e:
        print(f"[Auth Error] {e}")
        return "Unknown", 0.0


def authentication_system():
    """
    Runs the live-camera loop, matches faces against stored templates.

    Returns
    -------
    str  | None
        Username on success, None if authentication failed or was cancelled.
    """
    saved_db = load_embeddings()
    if not saved_db:
        print("No registered users found. Please register first.")
        return None

    cap = cv2.VideoCapture(0)

    consecutive         = 0
    last_user           = None
    authenticated_user  = None
    auth_start          = None
    printed_success_msg = False  # so we print only once

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, FACE_DETECTION_SCALE, FACE_DETECTION_NEIGHBORS,
            minSize=MIN_FACE_SIZE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = frame[y:y + h, x:x + w]

            user, sim = authenticate_user(roi, saved_db)
            cv2.putText(frame, f"Sim: {sim:.2f}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # running vote of consecutive matches
            if user != "Unknown":
                if user == last_user:
                    consecutive += 1
                else:
                    consecutive = 1
                    auth_start  = time.time()
                last_user = user

                if consecutive >= MIN_CONSECUTIVE_MATCHES:
                    authenticated_user = user
                    if not printed_success_msg:
                        print(f"Authentication successful – user: {user}")
                        printed_success_msg = True
            else:
                consecutive, last_user = 0, None
                if authenticated_user and time.time() - auth_start > TIMEOUT_SECONDS:
                    authenticated_user = None
                    printed_success_msg = False

            # overlay status label
            label = f"Access Granted: {authenticated_user}" if authenticated_user \
                    else "Access Denied"
            color = (0, 255, 0) if authenticated_user else (0, 0, 255)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("User Authentication System", frame)

        # return immediately when authenticated
        if authenticated_user:
            cap.release(); cv2.destroyAllWindows()
            return authenticated_user

        # quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    print("Authentication rejected")
    return None


if __name__ == "__main__":
    authentication_system()
