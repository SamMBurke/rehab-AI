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


# # Create a pose landmarker instance with the live stream mode:
# def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     print('pose landmarker result: {}'.format(result))

# options = PoseLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result)

# with PoseLandmarker.create_from_options(options) as landmarker:
#   # The landmarker is initialized. Use it here.
#   # ...

# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
# landmarker.detect_async(mp_image, frame_timestamp_ms)
