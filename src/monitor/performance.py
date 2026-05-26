import time
from collections import deque

import numpy as np


class PerformanceMonitor:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._latencies_ms = deque(maxlen=window_size)
        self._frame_times = deque(maxlen=window_size)

    def record_latency_ms(self, value: float) -> None:
        self._latencies_ms.append(value)
        self._frame_times.append(time.time())

    def get_avg_latency_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        return float(np.mean(self._latencies_ms))

    def get_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        duration = self._frame_times[-1] - self._frame_times[0]
        if duration <= 0:
            return 0.0
        return float((len(self._frame_times) - 1) / duration)
