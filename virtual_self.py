import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions

FPS = 30
DELAY_SECONDS = 3.1
DELAY_FRAMES = int(FPS * DELAY_SECONDS)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task")

detector = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
))

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def draw_landmarks(image, landmarks, mirrored, color_line, color_point):
    if not landmarks:
        return
    h, w = image.shape[:2]
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start, end = landmarks[start_idx], landmarks[end_idx]
            start_x = int((1.0 - start.x) * w) if mirrored else int(start.x * w)
            end_x = int((1.0 - end.x) * w) if mirrored else int(end.x * w)
            start_y, end_y = int(start.y * h), int(end.y * h)
            cv2.line(image, (start_x, start_y), (end_x, end_y), color_line, 2)
    for landmark in landmarks:
        x = int((1.0 - landmark.x) * w) if mirrored else int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 3, color_point, -1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

buffer = deque(maxlen=DELAY_FRAMES)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    w = frame.shape[1]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    detection_result = detector.detect_for_video(mp_image, int(frame_count * 1000 // FPS))
    
    you_canvas = np.zeros_like(frame)
    her_canvas = np.zeros_like(frame)
    
    if detection_result.pose_landmarks:
        buffer.append(detection_result)
        for pose_landmarks in detection_result.pose_landmarks:
            draw_landmarks(you_canvas, pose_landmarks, False, (255, 255, 255), (0, 0, 255))
    
    if len(buffer) == DELAY_FRAMES:
        for pose_landmarks in buffer[0].pose_landmarks:
            draw_landmarks(her_canvas, pose_landmarks, True, (255, 0, 255), (255, 255, 255))
    
    you_canvas[:, w//2 + 20:] = 0
    her_canvas[:, :w//2 - 20] = 0
    
    output = frame + you_canvas + her_canvas
    
    cv2.imshow("Virtual Dance Partner", output)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
detector.close()
