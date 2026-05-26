from typing import Dict, Tuple

import cv2


Point3D = Tuple[float, float, float]
KeypointMap = Dict[str, Point3D]
ColorBGR = Tuple[int, int, int]


class SkeletonDrawer:
    MINIMAL_CONNECTIONS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
    ]

    DEFAULT_CONNECTIONS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_heel"),
        ("right_knee", "right_heel"),
        ("left_heel", "left_foot_index"),
        ("right_heel", "right_foot_index"),
    ]

    UPPER_CONNECTIONS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    ]

    FULL_CONNECTIONS = [
        ("nose", "left_eye_inner"),
        ("left_eye_inner", "left_eye"),
        ("left_eye", "left_eye_outer"),
        ("left_eye_outer", "left_ear"),
        ("nose", "right_eye_inner"),
        ("right_eye_inner", "right_eye"),
        ("right_eye", "right_eye_outer"),
        ("right_eye_outer", "right_ear"),
        ("mouth_left", "mouth_right"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("left_wrist", "left_pinky"),
        ("left_wrist", "left_index"),
        ("left_wrist", "left_thumb"),
        ("left_pinky", "left_index"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("right_wrist", "right_pinky"),
        ("right_wrist", "right_index"),
        ("right_wrist", "right_thumb"),
        ("right_pinky", "right_index"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
        ("left_ankle", "left_heel"),
        ("right_ankle", "right_heel"),
        ("left_heel", "left_foot_index"),
        ("right_heel", "right_foot_index"),
        ("left_ankle", "left_foot_index"),
        ("right_ankle", "right_foot_index"),
    ]

    CONNECTIONS_BY_MODE = {
        "minimal": MINIMAL_CONNECTIONS,
        "default": DEFAULT_CONNECTIONS,
        "upper": UPPER_CONNECTIONS,
        "full": FULL_CONNECTIONS,
    }

    def __init__(self, skeleton_color: ColorBGR = (0, 255, 255), keypoint_mode: str = "default"):
        """Initialize drawer with skeleton color (BGR format).
        
        Default is yellow (0, 255, 255).
        Examples:
          - Red: (0, 0, 255)
          - Green: (0, 255, 0)
          - Blue: (255, 0, 0)
          - White: (255, 255, 255)
        """
        self.skeleton_color = skeleton_color
        self.connections = self.CONNECTIONS_BY_MODE.get(keypoint_mode, self.DEFAULT_CONNECTIONS)

    def draw(self, frame, keypoints: KeypointMap):
        h, w = frame.shape[:2]

        # Draw joints.
        for _, (x, y, _) in keypoints.items():
            cx = int(x * w)
            cy = int(y * h)
            cv2.circle(frame, (cx, cy), 5, self.skeleton_color, -1)

        # Draw bones.
        for start, end in self.connections:
            if start in keypoints and end in keypoints:
                x1, y1, _ = keypoints[start]
                x2, y2, _ = keypoints[end]
                cv2.line(
                    frame,
                    (int(x1 * w), int(y1 * h)),
                    (int(x2 * w), int(y2 * h)),
                    self.skeleton_color,
                    2,
                )

        return frame
