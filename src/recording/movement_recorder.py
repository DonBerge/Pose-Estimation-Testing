import json
import os
import time
from typing import Dict, Tuple


Point3D = Tuple[float, float, float]
KeypointMap = Dict[str, Point3D]


class MovementRecorder:
    """Stores skeleton motion as JSON Lines for later playback."""

    def __init__(self, output_path: str, metadata: dict):
        self.output_path = output_path
        self._start_time = time.perf_counter()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._fh = open(output_path, "w", encoding="utf-8")
        self._write_line({"type": "meta", **metadata})

    def record_frame(self, keypoints: KeypointMap) -> None:
        timestamp_s = time.perf_counter() - self._start_time
        serializable = {
            name: [float(v[0]), float(v[1]), float(v[2])] for name, v in keypoints.items()
        }
        self._write_line(
            {
                "type": "frame",
                "t": float(timestamp_s),
                "keypoints": serializable,
            }
        )

    def close(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def _write_line(self, payload: dict) -> None:
        self._fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
