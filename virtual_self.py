import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
import sys
import urllib.request
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions

MODEL_PATH = "pose_landmarker_heavy.task"

if getattr(sys, "frozen", False):
    MODEL_PATH = os.path.join(
        sys._MEIPASS,
        "pose_landmarker_heavy.task"
    )
else:
    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pose_landmarker_heavy.task"
    )

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


vid = cv2.VideoWriter('dance_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720))

DELAY_FRAMES = 15 
old = deque(maxlen=DELAY_FRAMES)

lwrist = deque(maxlen=20)
rwrist = deque(maxlen=20)

f = 0

# ================ This skeleton_bones connections is AI Assisted====================
# reference doc here - https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker

skeleton_bones = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),
    (26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]

# print(skeleton_bones)
#==============================================================================


def bones(image, landmarks, mirror, rgb_color):
    height, width = image.shape[:2]
    for connection in skeleton_bones:
        pt1, pt2 = connection

        # print(pt1, pt2)
        p1 = landmarks[pt1]
        p2 = landmarks[pt2]
        
        if mirror:
            x1 = int((1 - p1.x) * width)
            x2 = int((1 - p2.x) * width)
        else:
            x1 = int(p1.x * width)
            x2 = int(p2.x * width)

        # print (x1, x2)

        y1 = int(p1.y * height)
        y2 = int(p2.y * height)

        # print(y1, y2)

        cv2.line(image, (x1, y1), (x2, y2), rgb_color, 2)


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO
)

detector = PoseLandmarker.create_from_options(options)

# print("camra running")

while cam.isOpened():
    success, current_frame = cam.read()
    if not success:
        print("error")
        break

    
    current_frame = cv2.flip(current_frame, 1) 
    img_h, img_w = current_frame.shape[:2]


    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    
    timestamp_ms = int(f * 1000 / 30)
    detection_results = detector.detect_for_video(mp_img, timestamp_ms)
    

    skeleton_canvas = np.zeros_like(current_frame)

    if detection_results.pose_landmarks:
        
        single_pose = detection_results.pose_landmarks[0]
        old.append(single_pose)
        
        
        bones(skeleton_canvas, single_pose, mirror=False, rgb_color=(255, 255, 255))
        
        
        lw_x, lw_y = int(single_pose[15].x * img_w), int(single_pose[15].y * img_h)
        rw_x, rw_y = int(single_pose[16].x * img_w), int(single_pose[16].y * img_h)
        lwrist.append((lw_x, lw_y))
        rwrist.append((rw_x, rw_y))

    
    for idx in range(1, len(lwrist)):
        cv2.line(current_frame, lwrist[idx-1], lwrist[idx], (0, 255, 255), 4)
        
    for idx in range(1, len(rwrist)):
        cv2.line(current_frame, rwrist[idx-1], rwrist[idx], (255, 0, 255), 4)


    output = cv2.add(current_frame, skeleton_canvas)


    if len(old) == DELAY_FRAMES:
        ghost = old[0]
        bones(output, ghost, mirror=True, rgb_color=(50, 50, 50)) 
    
    cv2.imshow("Virtual Dance Partner", output)
    vid.write(output)

    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("exit")
        break
        
    f += 1


cam.release()
vid.release()
cv2.destroyAllWindows()
detector.close()
