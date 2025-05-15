"""
standalone camera_calibration.py
===============================
Calibrates a USB/web camera with a standard chessboard and stores the
intrinsics in **camera_calib.npz** (camera matrix K + distortion vector).

Quick start
-----------
1. **Print or display** a 10×7‐square chessboard (9×6 inner corners).
   • PDF generator: https://raw.githubusercontent.com/vanadiumlabs/calibration-patterns/master/chessboards/10x7_A4_25mm.pdf
2. Run the tool:
   ```bash
   python calibrate_camera_tool.py            # defaults cam 0, 15 shots
   # or tweak:
   python calibrate_camera_tool.py --cam 1 --shots 20 --out my_cam.npz
   ```
3. Move the board around; when green dots appear press **Space** to add a
   shot.  Collect the requested number, then press **Esc**.
4. The script prints the RMS reprojection error and writes the `.npz`.

Later you can load the file:
```python
import cv2, numpy as np
param = np.load('camera_calib.npz')
K, dist = param['mtx'], param['dist']
```

Tested with OpenCV 4.11.0 (has findChessboardCornersSB).
"""

from __future__ import annotations
import cv2, numpy as np, argparse, os, sys

# -----------------------------------------------------------------------------
# Default pattern and output
# -----------------------------------------------------------------------------
CHESSBOARD_SIZE   = (9, 6)      # inner corners (cols, rows)
SQUARE_SIZE_MM    = 24.0        # edge of one square (only scale, not used)
DEFAULT_OUT_FILE  = "camera_calib.npz"

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def detect_corners(gray: np.ndarray):
    """Try SB detector first; fall back to classic."""
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, CHESSBOARD_SIZE, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    else:
        found, corners = False, None
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    return found, corners

# -----------------------------------------------------------------------------
# Calibration capture loop
# -----------------------------------------------------------------------------

def interactive_capture(cam_id: int, required: int):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        sys.exit("❌ Could not open camera – check index or permissions")

    objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
    objp[:, :2] = np.indices(CHESSBOARD_SIZE).T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points, img_points = [], []
    print("» Show the 10×7 board, press SPACE to capture, ESC to quit «")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = detect_corners(gray)

        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, found)
        else:
            cv2.putText(vis, "Pattern not found", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis, f"Shots: {len(obj_points)}/{required}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Calibration", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc
            break
        if k == 32 and found:  # Space
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp.copy())
            img_points.append(corners.reshape(-1, 2))
            print(f"  ▸ Captured {len(obj_points)}/{required}")
            if len(obj_points) >= required:
                break

    cap.release(); cv2.destroyWindow("Calibration")
    return obj_points, img_points, gray.shape[::-1]

# -----------------------------------------------------------------------------
# Main calibration routine
# -----------------------------------------------------------------------------

def calibrate(cam_id: int, shots: int, out_file: str):
    obj_pts, img_pts, img_size = interactive_capture(cam_id, shots)
    if len(obj_pts) < 5:
        print("✖ Not enough good views – calibration aborted.")
        return False

    rms, K, dist, *_ = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
    np.savez_compressed(out_file, mtx=K, dist=dist)
    print(f"\n✔ Calibration complete. RMS error {rms:.3f} px")
    print(f"Saved → {out_file}\n")

    # quick preview of undistortion
    cap = cv2.VideoCapture(cam_id)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, img_size, cv2.CV_16SC2)
    print("Press ESC to close preview window…")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        stacked = np.hstack((frame, undist))
        cv2.imshow("Original  |  Undistorted", cv2.resize(stacked, None, fx=0.5, fy=0.5))
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release(); cv2.destroyAllWindows()
    return True

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Interactive webcam calibration (OpenCV 4.11)")
    p.add_argument("--cam", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--shots", type=int, default=15, help="number of good views to capture")
    p.add_argument("--out", default=DEFAULT_OUT_FILE, help="output .npz file")
    args = p.parse_args()

    calibrate(args.cam, args.shots, args.out)
