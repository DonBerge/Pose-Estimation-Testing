import argparse
import json
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.render.skeleton_drawer import SkeletonDrawer


Point3D = Tuple[float, float, float]
KeypointMap = Dict[str, Point3D]
ColorBGR = Tuple[int, int, int]


SKELETON_COLORS = {
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
}


def parse_color(color_str: str) -> ColorBGR:
    value = color_str.lower().strip()
    if value in SKELETON_COLORS:
        return SKELETON_COLORS[value]
    try:
        parts = value.split(",")
        if len(parts) == 3:
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return (b, g, r)
    except (ValueError, IndexError):
        pass
    print(f"Warning: invalid color '{color_str}', using yellow")
    return SKELETON_COLORS["yellow"]


def load_recording(path: str):
    meta = {}
    frames: List[dict] = []

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("type") == "meta":
                meta = payload
            elif payload.get("type") == "frame":
                keypoints = {
                    name: (float(values[0]), float(values[1]), float(values[2]))
                    for name, values in payload.get("keypoints", {}).items()
                }
                frames.append({"t": float(payload.get("t", 0.0)), "keypoints": keypoints})

    if not frames:
        raise RuntimeError("Recording has no frames")

    return meta, frames


def draw_info(frame, mode: str, speed: float, index: int, total: int):
    info = f"Replay | mode={mode} | speed={speed:.2f}x | frame={index + 1}/{total}"
    cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)


def replay(meta: dict, frames: List[dict], width: int, height: int, mode: str, color: ColorBGR, speed: float, loop: bool):
    drawer = SkeletonDrawer(skeleton_color=color, keypoint_mode=mode)

    while True:
        t0 = time.perf_counter()
        prev_t = frames[0]["t"]

        for i, frame_data in enumerate(frames):
            target_dt = (frame_data["t"] - prev_t) / max(speed, 1e-6)
            prev_t = frame_data["t"]

            # Sleep to preserve recorded timing profile.
            if i > 0 and target_dt > 0:
                elapsed = time.perf_counter() - t0
                wanted = frame_data["t"] / max(speed, 1e-6)
                remaining = wanted - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            output = drawer.draw(canvas, frame_data["keypoints"])
            draw_info(output, mode, speed, i, len(frames))
            cv2.imshow("Skeleton Replay", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return

        if not loop:
            break

    cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description="Replay skeleton movement recordings")
    parser.add_argument("--input", required=True, help="Input recording JSONL file")
    parser.add_argument("--width", type=int, default=0, help="Override output width")
    parser.add_argument("--height", type=int, default=0, help="Override output height")
    parser.add_argument(
        "--keypoint-mode",
        default=None,
        choices=["minimal", "default", "upper", "full"],
        help="Override keypoint mode",
    )
    parser.add_argument("--skeleton-color", default="yellow", help="Skeleton color")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    return parser


def main():
    args = build_parser().parse_args()
    meta, frames = load_recording(args.input)

    rec_size = meta.get("frame_size", [640, 480])
    width = args.width if args.width > 0 else int(rec_size[0])
    height = args.height if args.height > 0 else int(rec_size[1])
    mode = args.keypoint_mode if args.keypoint_mode else meta.get("keypoint_mode", "default")

    replay(
        meta=meta,
        frames=frames,
        width=width,
        height=height,
        mode=mode,
        color=parse_color(args.skeleton_color),
        speed=max(0.1, args.speed),
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
