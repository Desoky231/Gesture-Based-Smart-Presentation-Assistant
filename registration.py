"""
face_register_tf_modular.py
---------------------------
A **function-first** wrapper around the TensorFlow FaceNet enrolment
pipeline.  Import and call `register_user()` from any program, or run
this file directly for a small CLI.

Example (import):
-----------------
```python
from face_register_tf_modular import register_user

# prompt for name, capture 8 frames with Tan & Triggs, no GUI
register_user(frames=8, preproc="tantriggs", gui=False)
```

Example (CLI):
--------------
```bash
python face_register_tf_modular.py --user alice --frames 10 --preproc clahe
```

Dependencies
------------
* tensorflow >= 2.x
* mtcnn
* keras-facenet
* opencv-python   (opencv-python-headless also works if you keep `--nogui`)
"""

import os, argparse
from typing import List, Optional

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from keras_facenet import FaceNet

__all__ = [
    "register_user",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMB_DIR = "user_embeddings"
IMG_SIZE = 160  # FaceNet input size
os.makedirs(EMB_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Photometric pre-processing
# ---------------------------------------------------------------------------

def _safe_create_clahe():
    if hasattr(cv2, "createCLAHE"):
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    print("[WARN] OpenCV build lacks createCLAHE – CLAHE disabled.")
    return None

_CLAHE = _safe_create_clahe()


def _clahe(img_bgr: np.ndarray) -> np.ndarray:
    if _CLAHE is None:
        return img_bgr
    yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    yuv[:, :, 0] = _CLAHE.apply(yuv[:, :, 0])
    img_bgr = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)
    return cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=0)


def _tan_triggs(img_bgr: np.ndarray, gamma: float = 0.2,
                sigma0: float = 1.0, sigma1: float = 2.0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.power(gray, gamma)
    g0 = cv2.GaussianBlur(gray, (0, 0), sigma0)
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    tt = g0 - g1
    _, std = cv2.meanStdDev(tt)
    tt = tt / (std + 1e-8)
    tt = tt / (np.linalg.norm(tt, ord=1) + 1e-8)
    tt = np.tanh(0.1 * tt)
    tt = (tt - tt.min()) / (tt.max() - tt.min() + 1e-8)
    tt = (tt * 255).astype(np.uint8)
    return cv2.cvtColor(tt, cv2.COLOR_GRAY2BGR)


_PREPROC_FUNCS = {
    "clahe": _clahe,
    "tantriggs": _tan_triggs,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_and_crop(img: np.ndarray, box: List[int], target_size: int = IMG_SIZE,
                    margin: int = 20) -> Optional[np.ndarray]:
    x, y, w, h = box
    x = max(0, x - margin // 2)
    y = max(0, y - margin // 2)
    w += margin
    h += margin
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (target_size, target_size))


def _norm_pool(embeddings: List[np.ndarray]) -> np.ndarray:
    embs = np.stack(embeddings).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    return embs.mean(axis=0)

# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def register_user(user: Optional[str] = None,
                  camera: int = 0,
                  frames: int = 10,
                  preproc: str = "tantriggs",
                  gui: bool = True,
                  save_dir: str = EMB_DIR) -> Optional[np.ndarray]:
    """Capture *frames* crops, build a lighting-invariant FaceNet template,
    and save it under *save_dir* as `<user>.npz`.

    Parameters
    ----------
    user : str | None
        Username; if *None*, prompts via stdin.
    camera : int
        Index passed to `cv2.VideoCapture`.
    frames : int
        Number of successful face crops to capture.
    preproc : {"clahe", "tantriggs"}
        Photometric normaliser.
    gui : bool
        Whether to display live preview with bounding boxes.
    save_dir : str
        Directory for .npz templates.

    Returns
    -------
    np.ndarray | None
        The 512-D template, or *None* if capture failed.
    """
    if preproc == "clahe" and _CLAHE is None:
        print("[WARN] CLAHE unavailable – using Tan & Triggs instead.")
        preproc = "tantriggs"
    preprocess = _PREPROC_FUNCS[preproc]

    detector = MTCNN()
    embedder = FaceNet()

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera device")

    user = user or input("Type your user name: ").strip().lower()
    collected = []

    print("[INFO] Capture your face – turn slowly; press q to abort…")
    while len(collected) < frames:
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            faces = detector.detect_faces(frame)
        except (tf.errors.InvalidArgumentError, ValueError):
            continue  # malformed frame
        if not faces:
            if gui:
                cv2.putText(frame, "No face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Register", frame); cv2.waitKey(1)
            continue
        face = max(faces, key=lambda d: d['box'][2] * d['box'][3])
        crop = _align_and_crop(frame, face['box'])
        if crop is None:
            continue
        crop = preprocess(crop)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        emb1, emb2 = embedder.embeddings([crop_rgb, crop_rgb[:, ::-1]])
        collected.append((emb1 + emb2) / 2.0)

        if gui:
            cv2.putText(frame, f"Captured {len(collected)}/{frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Register", frame); cv2.waitKey(1)

    cap.release()
    if gui:
        cv2.destroyAllWindows()

    if len(collected) < 3:
        print("✖ Not enough good frames – please try again.")
        return None

    template = _norm_pool(collected)
    path = os.path.join(save_dir, f"{user}.npz")
    np.savez_compressed(path, embedding=template)
    print(f"✔ User '{user}' registered with {len(collected)} samples. Saved → {path}")
    return template

# ---------------------------------------------------------------------------
# Optional CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FaceNet enrolment CLI")
    p.add_argument("--user", help="username (file name)")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--preproc", choices=["clahe", "tantriggs"], default="tantriggs")
    p.add_argument("--nogui", action="store_true")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    register_user(user=args.user, frames=args.frames, preproc=args.preproc,
                  gui=not args.nogui, camera=args.camera)