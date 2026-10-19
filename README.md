# Virtual Dance Partner

An interactive Python script that uses webcam to create a mirrored skeleton of your movements. It uses Google's MediaPipe Pose detection to track your body and projects a delayed shadow next to you.

## Features
* **Pose Tracking:** Uses the MediaPipe Tasks Vision API for body landmark detection.
* **Delayed "Shadow":** Buffers your previous movements to project a mirrored version of yourself on the screen.
* **Auto-Downloading Model:** Automatically fetches the required `pose_landmarker_heavy.task` model file on the first run so you don't have to download it manually.

## Prerequisites

You need Python 3.7 or newer installed. You will also need to install the following Python libraries:

* `opencv-python` (cv2)
* `mediapipe`
* `numpy`

## Installation

1. Clone or download this repository/script to your local machine or install the exe file from the release section. 
2. Install the required dependencies using pip:

```bash
pip install opencv-python mediapipe numpy
