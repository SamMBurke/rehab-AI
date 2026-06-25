import numpy as np
import cv2 as cv
import mediapipe as mp

model_path = '../../models/pretrained/pose_landmarker_heavy.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


stream = cv.VideoCapture(0)

while stream.isOpened():
    success, frame = stream.read()
    cv.imshow("camera", frame)
    if cv.waitKey(1) == ord('q'):
        break

stream.release()
cv.destroyAllWindows()
