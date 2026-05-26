from dataclasses import dataclass
from typing import Tuple, Union


VideoSource = Union[int, str]
ColorBGR = Tuple[int, int, int]


@dataclass
class AppSettings:
    # Use 0 for local webcam, or a URL (RTSP/MJPEG/HTTP).
    source: VideoSource = 0

    # Lower resolution helps stability on CPU-only systems.
    frame_size: Tuple[int, int] = (640, 480)
    target_fps: float = 30.0

    # Pose confidence thresholds.
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    visibility_threshold: float = 0.5

    # Exponential smoothing factor for keypoints (higher = more stable, less reactive).
    smoothing_alpha: float = 0.85

    # Keypoint mode: minimal, default, upper, full.
    keypoint_mode: str = "default"

    # Pose model variant for MediaPipe tasks: lite, full, heavy.
    pose_model_variant: str = "heavy"

    # Stream robustness settings.
    reconnect_interval_s: float = 1.0
    read_timeout_s: float = 3.0

    # Performance settings.
    show_performance_overlay: bool = True

    # Skeleton color in BGR format (default: yellow).
    skeleton_color: ColorBGR = (0, 255, 255)


SETTINGS = AppSettings()
