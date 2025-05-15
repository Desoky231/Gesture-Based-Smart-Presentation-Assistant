"""
Gesture-Based Smart Presentation Assistant with GUI-based login
--------------------------------------------------------------
This refactor replaces the plain-console prompt with a very small
Tkinter dialog that lets the user either **Register** or **Authenticate**.
The meat of the original presentation/hand-tracking loop is wrapped in
`run_presentation()` so it is only entered if authentication succeeds.

* Register ➜ calls `register_user()` and shows a message.  User must then
  click **Authenticate** to log in (the program stays open, so no need to
  restart).
* Authenticate ➜ calls `authentication_system()`.  On success the GUI
  disappears and the OpenCV slide-control window starts.
  On failure the user is shown an error dialog and stays on the login
  screen; they cannot proceed until they succeed.

The rest of the code is exactly your original slide-handling logic,
just moved into its own function.

Dependencies:  Tkinter (built-in), cv2 (OpenCV), numpy, comtypes
(Windows-only PowerPoint automation).  Your existing helper modules
(`registration.py`, `authentication.py`, `HandTracker.py`,
`dottedline.py`) are used unchanged.
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import comtypes.client
import argparse                     
import pyttsx3                       # add once at top of your file
engine = pyttsx3.init()              # create once (long-lived)


# ---- Project-specific helpers ------------------------------------------------
from registration import register_user
from authentication import authentication_system
from HandTracker import HandDetector
from dottedline import drawrect, drawline

# ---- CONSTANTS ---------------------------------------------------------------
PPTX_PATH = r"C:\Users\alima\Downloads\Gesture-Based Smart Presentation Assistant\Dataanalysesproject.pptx"
OUTPUT_FOLDER = "ConvertedSlides"
FRAME_W, FRAME_H = 1280, 720  # camera resolution
HS, WS = int(120 * 1.2), int(213 * 1.2)  # thumbnail size
GE_THRESH_Y = 400
GE_THRESH_X = 750
GESTURE_DELAY_FRAMES = 15

# =============================================================================
# Helper: convert pptx to a folder of JPEG slides (PowerPoint → images)
# =============================================================================

def convert_pptx_to_images(pptx_path: str, output_folder: str) -> None:
    if not os.path.exists(pptx_path):
        raise FileNotFoundError(
            f"PowerPoint file not found at: {pptx_path}\n"
            "Please make sure the file exists and the path is correct."
        )

    os.makedirs(output_folder, exist_ok=True)

    try:
        powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
        powerpoint.Visible = 1
        presentation = powerpoint.Presentations.Open(pptx_path, WithWindow=False)
        # 17 = ppSaveAsJPG → each slide saved as Slide1.JPG, Slide2.JPG, …
        presentation.SaveAs(os.path.abspath(output_folder), 17)
        presentation.Close()
        powerpoint.Quit()
    except Exception as e:
        raise RuntimeError(
            "PowerPoint conversion failed. Please ensure PowerPoint is installed"
            " and that the file is not corrupted.\n" + str(e)
        ) from e


# =============================================================================
# Main presentation loop – unchanged logic, but wrapped in a function
# =============================================================================

# ---------------------------------------------------------------------------
# Camera calibration helpers
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Tunables (change once and the function picks them up automatically)
# ------------------------------------------------------------------------
CHESSBOARD_SIZE = (9, 6)        # inner corners (means 10x7 squares)
SQUARE_SIZE_MM  = 24.0
CALIB_FILE      = "camera_calib.npz"
FRAME_W, FRAME_H = 1280, 720

def calibrate_camera(cam_id=0, required=15):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
    objp[:, :2] = np.indices(CHESSBOARD_SIZE).T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_pts, img_pts = [], []
    print("Show the 10x7 chessboard, press SPACE to capture, ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(
            gray, CHESSBOARD_SIZE, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, CHESSBOARD_SIZE, corners, found)
        else:
            cv2.putText(view, "Pattern not found", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(view, f"Shots: {len(obj_pts)}/{required}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Calibration", view)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:                 # ESC
            break
        if k == 32 and found:       # SPACE
            obj_pts.append(objp.copy())
            img_pts.append(corners.squeeze())
            print(f" Captured {len(obj_pts)}/{required}")
            if len(obj_pts) >= required:
                break

    cap.release(); cv2.destroyWindow("Calibration")

    if len(obj_pts) < 5:
        print("Not enough views.")
        return False

    rms, K, dist, *_ = cv2.calibrateCamera(
        obj_pts, img_pts, (FRAME_W, FRAME_H), None, None)
    np.savez_compressed(CALIB_FILE, mtx=K, dist=dist)
    print(f"RMS = {rms:.3f}px  → saved to {CALIB_FILE}")
    return True


def load_calibration(file_path: str = CALIB_FILE):
    if not os.path.exists(file_path):
        return None, None
    data = np.load(file_path)
    mtx, dist = data["mtx"], data["dist"]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (FRAME_W, FRAME_H), 1, (FRAME_W, FRAME_H))
    map1, map2 = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (FRAME_W, FRAME_H), cv2.CV_16SC2)
    return map1, map2

# load maps once; if missing we’ll ask the user in GUI
MAP1, MAP2 = load_calibration()


def undistort(img: np.ndarray) -> np.ndarray:
    return cv2.remap(img, MAP1, MAP2, interpolation=cv2.INTER_LINEAR) if MAP1 is not None else img

# ---------------------------------------------------------------------------
# Presentation loop (unchanged logic except for undistort) -------------------
# ---------------------------------------------------------------------------

def run_presentation(presenter_name: str, tts) -> None:
    """
    Slide-show loop with hand-gesture control.

    Parameters
    ----------
    presenter_name : str
        Name printed on every slide (bottom left).
    tts : pyttsx3.Engine
        Text-to-speech engine; will speak short feedback phrases.
    """
    # ---------------- safety & slide loading -----------------------------
    if MAP1 is None:
        raise RuntimeError("Camera not calibrated – pick “Calibrate camera” first")

    if not os.listdir(OUTPUT_FOLDER):
        convert_pptx_to_images(PPTX_PATH, OUTPUT_FOLDER)

    slides = sorted([f for f in os.listdir(OUTPUT_FOLDER)
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not slides:
        raise FileNotFoundError("No slide images in ConvertedSlides/")

    # ---------------- state variables ------------------------------------
    slide_idx  = 0
    annotations = [[]]                 # list[list[(x,y)]]
    annot_idx   = 0
    gesture_on  = False
    gest_cd     = 0                    # cool-down counter
    pointer_on  = False
    feedback    = ""                   # text shown for 1 s
    feed_timer  = 0

    # ---------------- helper ---------------------------------------------
    def speak(msg: str):
        """Non-blocking TTS feedback."""
        tts.say(msg)
        tts.runAndWait()

    # ---------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = undistort(frame)
        frame = cv2.flip(frame, 1)

        # load current slide
        slide = cv2.imread(os.path.join(OUTPUT_FOLDER, slides[slide_idx]))
        slide = cv2.resize(slide, (FRAME_W, FRAME_H))

        # ---------------- hand detection ---------------------------------
        hands, frame = detector.findHands(frame)
        drawrect(frame, (FRAME_W, 0), (GE_THRESH_X, GE_THRESH_Y),
                 (0, 255, 0), 5, "dotted")

        if hands and not gesture_on:
            hand = hands[0]
            cx, cy = hand["center"]
            lm     = hand["lmList"]
            fingers = detector.fingersUp(hand)

            # fingertip mapped to slide space
            x_f = int(np.interp(lm[8][0], [FRAME_W//2, FRAME_W], [0, FRAME_W]))
            y_f = int(np.interp(lm[8][1], [150, FRAME_H-150], [0, FRAME_H]))
            tip  = (x_f, y_f)

            # --- inside gesture box ----------------------------------
            if cy < GE_THRESH_Y and cx > GE_THRESH_X:
                if fingers == [1,0,0,0,0] and slide_idx > 0:
                    slide_idx -= 1
                    annotations = [[]]; annot_idx = 0
                    gesture_on, feedback = True, "Previous Slide"
                    speak("Previous slide")
                elif fingers == [0,0,0,0,1] and slide_idx < len(slides)-1:
                    slide_idx += 1
                    annotations = [[]]; annot_idx = 0
                    gesture_on, feedback = True, "Next Slide"
                    speak("Next slide")
                elif fingers == [1,1,1,1,1]:
                    annotations = [[]]; annot_idx = 0
                    gesture_on, feedback = True, "Cleared"
                    speak("Clearing annotations")

            # --- pointer & draw --------------------------------------
            if fingers == [0,1,1,0,0]:           # pointer
                cv2.circle(slide, tip, 5, (0,0,255), cv2.FILLED)
            elif fingers == [0,1,0,0,0]:         # draw
                if len(annotations)==0: annotations.append([])
                if tip not in annotations[annot_idx]:
                    annotations[annot_idx].append(tip)
                    cv2.circle(slide, tip, 5, (0,0,255), cv2.FILLED)
            elif fingers == [0,1,1,1,0]:         # erase last stroke
                if annotations and annotations[-1]:
                    annotations.pop(); annot_idx = max(0, annot_idx-1)
                    gesture_on, feedback = True, "Undo"
                    speak("Erase last annotation")

        # ---------- one-gesture cool-down -------------------------------
        if gesture_on:
            gest_cd += 1
            if gest_cd > GESTURE_DELAY_FRAMES:
                gest_cd, gesture_on = 0, False
                feed_timer = 30                 # show feedback ~1 s

        # ---------- draw stored annotations -----------------------------
        for stroke in annotations:
            for i in range(1, len(stroke)):
                cv2.line(slide, stroke[i-1], stroke[i], (0,0,255), 6)

        # feedback text (fade after 1 s)
        if feed_timer > 0:
            cv2.putText(slide, f"Gesture: {feedback}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
            feed_timer -= 1

        # presenter name (bottom-left)
        cv2.putText(slide, f"Presenter: {presenter_name}",
                    (20, FRAME_H-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        # PiP webcam
        thumb = cv2.resize(frame, (WS, HS))
        slide[FRAME_H-HS:FRAME_H, FRAME_W-WS:FRAME_W] = thumb

        cv2.imshow("Slides", slide)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Tk front‑end with calibration option
# ---------------------------------------------------------------------------

def launch_gui():
    root = tk.Tk(); root.title("Hand‑Gesture Presentation"); root.geometry("320x220")
    tk.Label(root, text="Welcome – choose an option", font=("Arial", 12)).pack(pady=12)

    var = tk.StringVar(value="authenticate")
    tk.Radiobutton(root, text="Register new user", variable=var, value="register").pack(anchor="w", padx=40)
    tk.Radiobutton(root, text="Authenticate & start", variable=var, value="authenticate").pack(anchor="w", padx=40)
    tk.Radiobutton(root, text="Calibrate camera", variable=var, value="calibrate").pack(anchor="w", padx=40)

    def proceed():
        choice = var.get()
        if choice == "register":
            register_user()
            messagebox.showinfo("Registration", "Registration complete. Now authenticate to start the presentation.")
        elif choice == "calibrate":
            if calibrate_camera():
                global MAP1, MAP2
                MAP1, MAP2 = load_calibration()
                messagebox.showinfo("Calibration", "Calibration successful! You can now authenticate and start.")
        else:  # authenticate
            user = authentication_system()
            if user:
                
                try:
                    run_presentation(presenter_name=user, tts=engine)
                except RuntimeError as e:
                    messagebox.showerror("Error", str(e))
            else:
                messagebox.showerror("Authentication failed", "Invalid credentials or canceled. Please try again.")

    tk.Button(root, text="Continue", width=14, command=proceed).pack(pady=18)
    root.mainloop()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="run calibration only and exit")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_camera()
    else:
        try:
            launch_gui()
        except Exception as e:
            messagebox.showerror("Fatal error", str(e)); sys.exit(1)
