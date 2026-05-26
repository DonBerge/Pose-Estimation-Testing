import argparse
import time

import cv2

from src.config.settings import SETTINGS, AppSettings, ColorBGR
from src.monitor.performance import PerformanceMonitor
from src.pose.detector import PoseDetector
from src.pose.skeleton_processor import SkeletonProcessor
from src.render.skeleton_drawer import SkeletonDrawer
from src.recording.movement_recorder import MovementRecorder
from src.video.stream_handler import StreamHandler


# Predefined skeleton colors (BGR format).
SKELETON_COLORS = {
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
}


def parse_source(raw: str):
    if raw.isdigit():
        return int(raw)
    return raw


def parse_color(color_str: str) -> ColorBGR:
    """Parse color from name or 'R,G,B' format."""
    color_str = color_str.lower().strip()
    
    # Try predefined color names.
    if color_str in SKELETON_COLORS:
        return SKELETON_COLORS[color_str]
    
    # Try RGB format (e.g., "255,0,0").
    try:
        parts = color_str.split(",")
        if len(parts) == 3:
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return (b, g, r)  # Reverse to BGR.
    except (ValueError, IndexError):
        pass
    
    # Default to yellow if parsing fails.
    print(f"Warning: invalid color '{color_str}', using yellow")
    return SKELETON_COLORS["yellow"]


def build_settings(args) -> AppSettings:
    settings = AppSettings(
        source=parse_source(args.source),
        frame_size=(args.width, args.height),
        target_fps=args.fps,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        visibility_threshold=args.visibility_threshold,
        smoothing_alpha=args.smoothing_alpha,
        keypoint_mode=args.keypoint_mode,
        pose_model_variant=args.pose_model,
        reconnect_interval_s=args.reconnect_interval_s,
        read_timeout_s=args.read_timeout_s,
        show_performance_overlay=not args.no_perf_overlay,
        skeleton_color=parse_color(args.skeleton_color),
    )
    return settings


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Taekwondo pose estimation in real time")
    parser.add_argument("--source", default=str(SETTINGS.source), help="0 for webcam or URL for stream")
    parser.add_argument("--width", type=int, default=SETTINGS.frame_size[0])
    parser.add_argument("--height", type=int, default=SETTINGS.frame_size[1])
    parser.add_argument("--fps", type=float, default=SETTINGS.target_fps)
    parser.add_argument("--min-detection-confidence", type=float, default=SETTINGS.min_detection_confidence)
    parser.add_argument("--min-tracking-confidence", type=float, default=SETTINGS.min_tracking_confidence)
    parser.add_argument("--visibility-threshold", type=float, default=SETTINGS.visibility_threshold)
    parser.add_argument("--smoothing-alpha", type=float, default=SETTINGS.smoothing_alpha)
    parser.add_argument(
        "--keypoint-mode",
        default=SETTINGS.keypoint_mode,
        choices=["minimal", "default", "upper", "full"],
        help="Cantidad de keypoints a mostrar",
    )
    parser.add_argument(
        "--pose-model",
        default=SETTINGS.pose_model_variant,
        choices=["lite", "full", "heavy"],
        help="MediaPipe pose model variant",
    )
    parser.add_argument("--reconnect-interval-s", type=float, default=SETTINGS.reconnect_interval_s)
    parser.add_argument("--read-timeout-s", type=float, default=SETTINGS.read_timeout_s)
    parser.add_argument(
        "--skeleton-color",
        default="yellow",
        help="Skeleton color: yellow, red, green, blue, white, cyan, magenta, or R,G,B (0-255)",
    )
    parser.add_argument(
        "--record-movements",
        default="",
        help="Output JSONL file to save skeleton movement frames",
    )
    parser.add_argument("--no-perf-overlay", action="store_true")
    return parser


def draw_overlay(frame, monitor: PerformanceMonitor, stale: bool):
    text = f"FPS: {monitor.get_fps():.1f} | Latency: {monitor.get_avg_latency_ms():.1f}ms"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    status = "STREAM STALE" if stale else "TRACKING"
    color = (0, 0, 255) if stale else (0, 255, 0)
    cv2.putText(frame, status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    args = build_arg_parser().parse_args()
    settings = build_settings(args)

    stream = StreamHandler(settings.source, settings)
    detector = PoseDetector(settings)
    processor = SkeletonProcessor(settings)
    drawer = SkeletonDrawer(skeleton_color=settings.skeleton_color, keypoint_mode=settings.keypoint_mode)
    monitor = PerformanceMonitor()
    recorder = None

    if args.record_movements:
        recorder = MovementRecorder(
            output_path=args.record_movements,
            metadata={
                "version": 1,
                "source": str(settings.source),
                "frame_size": [settings.frame_size[0], settings.frame_size[1]],
                "target_fps": float(settings.target_fps),
                "keypoint_mode": settings.keypoint_mode,
                "pose_model": settings.pose_model_variant,
            },
        )
        print(f"Recording skeleton movements to {args.record_movements}")

    stream.start()

    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            results = detector.detect(frame)
            keypoints = processor.extract(results)
            keypoints = processor.smooth(keypoints)

            if recorder is not None:
                recorder.record_frame(keypoints)

            output = drawer.draw(frame, keypoints)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            monitor.record_latency_ms(latency_ms)

            if settings.show_performance_overlay:
                draw_overlay(output, monitor, stream.is_stale())

            cv2.imshow("Taekwondo Pose Estimation", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        stream.stop()
        detector.close()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
