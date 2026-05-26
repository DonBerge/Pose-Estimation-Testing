import threading
import time
from typing import Optional, Union

import cv2
import numpy as np

from src.config.settings import AppSettings


VideoSource = Union[int, str]


class StreamHandler:
    """Reads frames from a source and keeps only the freshest one.

    This design prioritizes stability and low display lag on CPU-only setups by
    dropping stale frames when processing cannot keep up.
    """

    def __init__(self, source: VideoSource, settings: AppSettings):
        self.source = source
        self.settings = settings

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_time = 0.0
        self._consecutive_failures = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._release_capture()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def is_stale(self) -> bool:
        return (time.time() - self._last_frame_time) > self.settings.read_timeout_s

    def _capture_loop(self) -> None:
        while self._running:
            if not self._ensure_capture_open():
                time.sleep(self.settings.reconnect_interval_s)
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 10:
                    self._release_capture()
                    time.sleep(self.settings.reconnect_interval_s)
                continue

            self._consecutive_failures = 0
            resized = cv2.resize(frame, self.settings.frame_size)
            with self._lock:
                self._latest_frame = resized
                self._last_frame_time = time.time()

    def _ensure_capture_open(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True

        self._release_capture()
        self._cap = cv2.VideoCapture(self.source)

        if self._cap is None or not self._cap.isOpened():
            self._release_capture()
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_size[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.settings.target_fps)
        return True

    def _release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
