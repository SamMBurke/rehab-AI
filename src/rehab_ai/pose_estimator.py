import numpy as np
import cv2 as cv
import mediapipe as mp

from rehab_ai.pose import Pose
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseEstimator():
    def __init__(self, model_path, running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM):
        self.model_path = model_path
        self.running_mode = running_mode

        # self._BaseOptions = mp.tasks.BaseOptions
        # self._PoseLandmarker = mp.tasks.vision.PoseLandmarker
        # self._PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        self._options = vision.PoseLandmarkerOptions(
           base_options = python.BaseOptions(model_asset_path = self.model_path),
           running_mode = self.running_mode,
           result_callback = self._result_callback
        )

        self._detector = vision.PoseLandmarker.create_from_options(self._options)
        self._latest_result = None

    def _result_callback(self, result, output_image, timestamp_ms):
        '''
        necessary callback function when doing live video input as it does the pose estimation asynchronously
        '''
        self._latest_result = result

    def detect_pose(self, frame, timestamp_ms):
        '''
        detects the pose landmarks in a given frame
        NOTE: timestamp_ms is a value from the OpenCV library an use for the asynchronous pose estimation
        '''
        if self.running_mode == mp.tasks.vision.RunningMode.LIVE_STREAM:
            self._detector.detect_async(frame, timestamp_ms)
            if self._latest_result is None:
                return None
            return Pose(self._latest_result)
        else:
            detection_result = self._detector.detect(frame)
            return Pose(detection_result)
