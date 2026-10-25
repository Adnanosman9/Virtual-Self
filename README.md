# virtual partner

This is a python script that makes clone skeleton version of yourself beside you that repeats your movement. MediaPipe api is used to track your body movement and repeat it. 

## 🎥 Quick Demo of It Working

<img width="426" height="240" alt="1023" src="https://github.com/user-attachments/assets/d82aa85a-a243-4b34-8501-f7b7569d645c" />

## Updated features
* Hand wrists have a glowing trail moving
* The video will be automatically recorded and saved in the folder

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

---

## Environment setup for MAC

Here is an error happens because Python installations on macOS (specifically those downloaded from python.org) do not automatically include SSL root certificates.

### Solve it manually - 
* Open Finder and go to the Applications folder.
* Open their Python 3.x folder (e.g., Python 3.10 or 3.11).
* Double-click the file named Install Certificates.command. A terminal will open, install the certificates, and fix the issue permanently for their machine.


## Debugging journey

### To track the body and make the skeleton, needs a skeleton connection. In Google's [Documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model), they gave this skeleton

<img width="676" height="724" alt="Screenshot 2026-10-23 184456" src="https://github.com/user-attachments/assets/3a4d0d87-0fe7-4000-979d-3a20feab0d6d" />

### Based on this skeleton, I was struggling to produce the proper connection metrics.

### Another problem was making the `.exe` including `pose_landmarker_heavy.task`. The .exe file was trying to find pose_landmarker in relative path folders. 

> ### These both issues were debugged with assistance of Gemini cause these were totally new problems for me. 



## Troubleshooting
* If the lines aren't tracking your arms or legs, step further back from your camera. MediaPipe needs to see your upper torso clearly to map out your shoulder and hip joints.
* Close down other active apps that might be hogging your video feed like Discord, Zoom, or open browser tabs.
* **Also check if the code is trying to connect to your other connected devices.**

License: MIT
