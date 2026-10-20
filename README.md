# virtual partner

This is a python script that makes clone skeleton version of yourself beside you that repeats your movement. MediaPipe api is used to track your body movement and repeat it. 

## 🎥 Quick Demo of It Working

<img width="1111" height="311" alt="Screenshot 2026-10-20 035233" src="https://github.com/user-attachments/assets/f2773c80-85ce-4c97-9ec6-1660188704d1" />

<img width="408" height="240" alt="1020 (2)" src="https://github.com/user-attachments/assets/d39a2b56-dc7a-408b-9e72-2bb1f543f3f7" />

## How to Run It

If you just want to run the code directly, follow these quick steps:

1. GO to my [Latest Release](https://github.com/Adnanosman9/Virtual-Self/releases/tag/V1) page and download the compiled `.exe` file.
2. You have to be connected with internet to run it the first time. The program will automatically download the machine learning model file (`pose_landmarker_heavy.task`) right into your project folder.
3. If you want to run the Python script instead, install the requirements manually:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
4. Then just boot up the script file:
   ```bash
   python virtual_self.py

   ```


## The Problems I had during development

I used OpenCV to hook into the webcam hardware and pull widescreen video frames. To handle the skeleton lines, I integrated Google's MediaPipe Pose Landmarker which tracks different joints on your body.

Installing third-party libraries every time is a time waste, so I set up a function that uses Python's built-in `urllib.request` and downloads the required ML model file automatic. 


## Troubleshooting
* If the lines aren't tracking your arms or legs, step further back from your camera. MediaPipe needs to see your upper torso clearly to map out your shoulder and hip joints.
* Close down other active apps that might be hogging your video feed like Discord, Zoom, or open browser tabs.
* **Also check if the code is trying to connect to your other connected devices.**

License: MIT
