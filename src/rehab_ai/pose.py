import mediapipe as mp
import numpy as np

class Pose():
    def __init__(self, detection_result):
        self.landmarks_list = detection_result.pose_landmarks

        self._DrawingStyles = mp.tasks.vision.drawing_styles
        self._DrawingUtils = mp.tasks.vision.drawing_utils

    def draw_landmarks(self, frame):
        annotated_image = np.copy(frame)

        pose_landmark_style = self._DrawingStyles.get_default_pose_landmarks_style()
        pose_connection_style = self._DrawingUtils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for landmarks in self.landmarks_list:
            self._DrawingUtils.draw_landmarks(
                image = annotated_image,
                landmark_list = landmarks,
                connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec = pose_landmark_style,
                connection_drawing_spec = pose_connection_style
            )
        return annotated_image