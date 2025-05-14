import cv2
import os
from registration import register_user
from authentication import authentication_system
from HandTracker import HandDetector
from dottedline import drawrect, drawline
import numpy as np
import comtypes.client
from utils import convert_pptx_to_images
import tkinter as tk
from tkinter import messagebox


# ========== SETTINGS ==========
pptx_path = r"C:\Users\abdel\Desktop\disk D\Undergrad\Semester 8\computer vision\project\Gesture-Based Smart Presentation Assistant\Gesture-Based Smart Presentation Assistant\Dataanalysesproject.pptx"
output_folder = "ConvertedSlides"
width, height = 1280, 720
hs, ws = int(120 * 1.2), int(213 * 1.2)
ge_thresh_y = 400
ge_thresh_x = 750
delay = 15

# ========== USER LOGIN SYSTEM ==========

# Create the main window
root = tk.Tk()
root.title("Hand Gesture Presentation System")
root.geometry("350x170") # Adjusted size for better text fit

# Create a label
label = tk.Label(root, text="Welcome to the Hand Gesture Presentation System", wraplength=330, justify="center")
label.pack(pady=10)

# Define button callbacks
def on_register_click():
    register_user()
    # After registration, the original script advised restarting.
    # Here, we'll close the window, and the user would manually restart if needed.
    root.destroy()

def on_authenticate_click():
    access_granted = authentication_system() # This calls your actual authentication logic
    if access_granted:
        messagebox.showinfo("Success", "Authentication successful. Starting presentation...")
        # Proceed to presentation logic here (e.g., open a new window or start the main app)
        root.destroy() # Close the login window
    else:
        messagebox.showerror("Error", "Authentication failed or canceled. Exiting...")
        root.destroy() # Close the login window

# Create buttons
btn_register = tk.Button(root, text="Register a new user", command=on_register_click)
btn_register.pack(pady=5, fill=tk.X, padx=20)

btn_authenticate = tk.Button(root, text="Authenticate to start the presentation", command=on_authenticate_click)
btn_authenticate.pack(pady=5, fill=tk.X, padx=20)

# Start the GUI event loop
root.mainloop()



try:
    convert_pptx_to_images(pptx_path, output_folder)
except Exception as e:
    print(f"\nError: {str(e)}")
    print("\nPlease fix the above issue and run the program again.")
    exit(1)

# ========== INIT ==========
slide_num = 0
gest_done = False
gest_counter = 0
annotations = [[]]
annot_num = 0
annot_start = False

# Load converted images
frames_folder = output_folder
path_imgs = sorted([img for img in os.listdir(frames_folder) if img.lower().endswith(('.jpg', '.png', '.jpeg'))], key=len)
print("Slides found:", path_imgs)

# ✅ Safety check
if not path_imgs:
    raise FileNotFoundError(f"No slides were found in: {frames_folder}. Check your PowerPoint path or conversion step.")

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = HandDetector(detectionCon=0.8, maxHands=1)


while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    pathFullImage = os.path.join(frames_folder, path_imgs[slide_num])
    slide_current = cv2.imread(pathFullImage)
    slide_current = cv2.resize(slide_current, (1280, 720)) 

    hands, frame = detector.findHands(frame)
    drawrect(frame, (width, 0), (ge_thresh_x, ge_thresh_y), (0, 255, 0), 5, 'dotted')

    if hands and not gest_done:
        hand = hands[0]
        cx, cy = hand["center"]
        lm_list = hand["lmList"]
        fingers = detector.fingersUp(hand)

        x_val = int(np.interp(lm_list[8][0], [width // 2, width], [0, width]))
        y_val = int(np.interp(lm_list[8][1], [150, height - 150], [0, height]))
        index_fing = x_val, y_val

        if cy < ge_thresh_y and cx > ge_thresh_x:
            annot_start = False

            if fingers == [1, 0, 0, 0, 0]:  # prev
                if slide_num > 0:
                    gest_done = True
                    slide_num -= 1
                    annotations = [[]]
                    annot_num = 0

            elif fingers == [0, 0, 0, 0, 1]:  # next
                if slide_num < len(path_imgs) - 1:
                    gest_done = True
                    slide_num += 1
                    annotations = [[]]
                    annot_num = 0

            elif fingers == [1, 1, 1, 1, 1]:  # clear
                annotations.clear()
                annot_num = 0
                gest_done = True
                annotations = [[]]

        if fingers == [0, 1, 1, 0, 0]:  # show pointer
            cv2.circle(slide_current, index_fing, 4, (0, 0, 255), cv2.FILLED)
            annot_start = False

        elif fingers == [0, 1, 0, 0, 0]:  # draw
            if not annot_start:
                annot_start = True
                annot_num += 1
                annotations.append([])
            annotations[annot_num].append(index_fing)
            cv2.circle(slide_current, index_fing, 4, (0, 0, 255), cv2.FILLED)

        else:
            annot_start = False

        if fingers == [0, 1, 1, 1, 0]:  # erase last
            if annotations and annot_num >= 0:
                annotations.pop(-1)
                annot_num -= 1
                gest_done = True
    else:
        annot_start = False

    if gest_done:
        gest_counter += 1
        if gest_counter > delay:
            gest_counter = 0
            gest_done = False

    for annotation in annotations:
        for j in range(1, len(annotation)):
            cv2.line(slide_current, annotation[j - 1], annotation[j], (0, 0, 255), 6)

    img_small = cv2.resize(frame, (ws, hs))
    h, w, _ = slide_current.shape
    slide_current[h - hs:h, w - ws:w] = img_small

    cv2.imshow("Slides", slide_current)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
