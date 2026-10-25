# Main detection imports
import cv2
import mediapipe as mp
import numpy as np
import time

# Imports for window closing and termination
import keyboard as key
import os

# Kill the script if a specific keybind happens
def KillScript():
  os._exit(0)

key.add_hotkey('ctrl+alt+p', KillScript)

# Import our custom One-Euro filter
from EuroFilter import OneEuroFilter


# ==========================================
#               CONFIGURATION
# ==========================================

# --- Tracking Stability & Certainty Settings ---
MIN_DETECTION_CONFIDENCE = 0.25
MIN_PRESENCE_CONFIDENCE = 0.3

OVERLAY_IMAGE_PATH = "TargetImage.png"     # Path to your custom overlay image
MODEL_PATH = "face_landmarker.task"  # MediaPipe face landmarker model file
BOX_SCALE_FACTOR = 1.6              # Size multiplier for the face bounding box

# --- Smoothing Settings (One-Euro Filter) ---
FILTER_MIN_CUTOFF = 0.8              # Lower = smoother when holding still (reduces jitter)
FILTER_BETA = 0.1                   # Higher = faster reaction / less lag during fast movements

# --- Tracking Stability Settings ---
MAX_MISSING_FRAMES = 20              # Frames to hold position if face is briefly lost (e.g. head turn)

# --- Display Settings ---
STRETCH_IMAGE = False         
# True = Stretch overlay to fit box dimensions (may distort)
# False = Maintain overlay's original aspect ratio (centers it)

# ==========================================

# Load custom overlay image
overlay_img = cv2.imread(OVERLAY_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
if overlay_img is None:
  print(
      f"Warning: Could not load '{OVERLAY_IMAGE_PATH}'. Falling back to solid"
      " black box."
  )

# Initialize MediaPipe Face Landmarker in VIDEO mode for optimal temporal stability
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,  # Uses continuous video tracking instead of frame-by-frame guessing
    num_faces=1,
    min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_face_presence_confidence=MIN_PRESENCE_CONFIDENCE,
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Filter instances
filter_x, filter_y, filter_w, filter_h = None, None, None, None
smooth_x, smooth_y, smooth_w, smooth_h = None, None, None, None

frame_count = 0

with FaceLandmarker.create_from_options(options) as landmarker:
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame_count += 1
    # Run MediaPipe inference only on every 2nd frame (cuts CPU load in half)
    run_detection = (frame_count % 2 == 0)

    current_time = time.time()
    timestamp_ms = int(current_time * 1000)
    
    h, w, _ = frame.shape
    face_detected_this_frame = False

    # Only run the heavy MediaPipe detection on alternating frames
    if run_detection:
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
      detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

      if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        
        xs = [landmark.x * w for landmark in landmarks]
        ys = [landmark.y * h for landmark in landmarks]
        
        raw_x = int(min(xs))
        raw_y = int(min(ys))
        raw_w = int(max(xs) - min(xs))
        raw_h = int(max(ys) - min(ys))

        expand_w = int(raw_w * (BOX_SCALE_FACTOR - 1) / 2)
        expand_h = int(raw_h * (BOX_SCALE_FACTOR - 1) / 2)

        target_x = max(0, raw_x - expand_w)
        target_y = max(0, raw_y - expand_h)
        target_w = min(w - target_x, raw_w + (expand_w * 2))
        target_h = min(h - target_y, raw_h + (expand_h * 2))

        if filter_x is None:
          filter_x = OneEuroFilter(current_time, target_x, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
          filter_y = OneEuroFilter(current_time, target_y, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
          filter_w = OneEuroFilter(current_time, target_w, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
          filter_h = OneEuroFilter(current_time, target_h, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
        
        smooth_x = int(filter_x(current_time, target_x))
        smooth_y = int(filter_y(current_time, target_y))
        smooth_w = int(filter_w(current_time, target_w))
        smooth_h = int(filter_h(current_time, target_h))

        face_detected_this_frame = True
        frames_since_last_detection = 0

    if not face_detected_this_frame and run_detection:
      frames_since_last_detection += 1
      if frames_since_last_detection > MAX_MISSING_FRAMES:
        filter_x, filter_y, filter_w, filter_h = None, None, None, None
        smooth_x, smooth_y, smooth_w, smooth_h = None, None, None, None

    # Render overlay using the cached/smoothed coordinates on every frame
    if smooth_x is not None and smooth_w > 0 and smooth_h > 0:
      if not STRETCH_IMAGE and overlay_img is not None:
        img_h, img_w = overlay_img.shape[:2]
        scale = min(smooth_w / img_w, smooth_h / img_h)
        draw_w = int(img_w * scale)
        draw_h = int(img_h * scale)
        draw_x = smooth_x + (smooth_w - draw_w) // 2
        draw_y = smooth_y + (smooth_h - draw_h) // 2
      else:
        draw_x = smooth_x
        draw_y = smooth_y
        draw_w = smooth_w
        draw_h = smooth_h

      x_end = min(w, draw_x + draw_w)
      y_end = min(h, draw_y + draw_h)
      region_w = x_end - draw_x
      region_h = y_end - draw_y

      if region_w > 0 and region_h > 0:
        if overlay_img is not None:
          resized_overlay = cv2.resize(
              overlay_img, (region_w, region_h), interpolation=cv2.INTER_AREA
          )
          if resized_overlay.shape[2] == 4:
            alpha = resized_overlay[:, :, 3:4] / 255.0
            overlay_rgb = resized_overlay[:, :, :3]
            roi = frame[draw_y:y_end, draw_x:x_end]
            
            frame[draw_y:y_end, draw_x:x_end] = (
                alpha * overlay_rgb + (1 - alpha) * roi
            ).astype(np.uint8)
          else:
            frame[draw_y:y_end, draw_x:x_end] = resized_overlay[:, :, :3]
        else:
          cv2.rectangle(
              frame, (draw_x, draw_y), (x_end, y_end), (0, 0, 0), -1
          )

    cv2.imshow("VHS Censor Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()