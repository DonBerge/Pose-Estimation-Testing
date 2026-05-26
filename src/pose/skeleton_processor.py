from typing import Dict, Tuple

import numpy as np

from src.config.settings import AppSettings


Point3D = Tuple[float, float, float]
KeypointMap = Dict[str, Point3D]


class SkeletonProcessor:
    # Minimal mode: torso + elbows + knees.
    MINIMAL_KEYPOINTS = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
    }

    # Default mode: current project set.
    DEFAULT_KEYPOINTS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }

    # Extra recommended mode for upper-body focused use-cases.
    UPPER_KEYPOINTS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
    }

    LANDMARK_NAMES = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]
    FULL_KEYPOINTS = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}

    KEYPOINTS_BY_MODE = {
        "minimal": MINIMAL_KEYPOINTS,
        "default": DEFAULT_KEYPOINTS,
        "upper": UPPER_KEYPOINTS,
        "full": FULL_KEYPOINTS,
    }

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._prev: KeypointMap = {}
        self._keypoints = self.KEYPOINTS_BY_MODE.get(settings.keypoint_mode, self.DEFAULT_KEYPOINTS)

    def extract(self, pose_results) -> KeypointMap:
        """Extract keypoints from detection results for the current frame only."""
        if pose_results.pose_landmarks is None:
            return {}

        landmarks = pose_results.pose_landmarks.landmark
        out: KeypointMap = {}
        for name, idx in self._keypoints.items():
            lm = landmarks[idx]
            if lm.visibility >= self.settings.visibility_threshold:
                out[name] = (lm.x, lm.y, lm.z)

        return out

    def smooth(self, keypoints: KeypointMap) -> KeypointMap:
        """Apply temporal smoothing to stabilize keypoint coordinates."""
        if not keypoints:
            return {}

        # First time or no history: just initialize.
        if not self._prev:
            self._prev = {name: np.array(pos, dtype=np.float32) for name, pos in keypoints.items()}
            return keypoints

        alpha = self.settings.smoothing_alpha
        smoothed: KeypointMap = {}

        # Smooth all keypoints from current detection.
        for name, current in keypoints.items():
            if name not in self._prev:
                # New keypoint that wasn't visible before; initialize it.
                smoothed[name] = current
                self._prev[name] = np.array(current, dtype=np.float32)
            else:
                # Apply exponential smoothing: high alpha means trust current more.
                prev = self._prev[name]
                cur = np.array(current, dtype=np.float32)
                value = alpha * cur + (1.0 - alpha) * prev
                smoothed[name] = (float(value[0]), float(value[1]), float(value[2]))
                self._prev[name] = value

        return smoothed
