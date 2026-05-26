import cv2
import os
import zipfile
import urllib.request
import urllib.error
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config.settings import AppSettings


# Model variants available in MediaPipe assets.
MODEL_VARIANTS = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}
MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
MODEL_DIR = "/tmp"


def _model_url(variant: str, model_name: str) -> str:
    return f"{MODEL_BASE_URL}/pose_landmarker_{variant}/float16/latest/{model_name}"


def _is_valid_model_file(path: str) -> bool:
    """MediaPipe .task is a zip container; validate before use."""
    if not os.path.exists(path):
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # testzip returns first bad file name or None when archive is fine.
            return zf.testzip() is None
    except Exception:
        return False


def ensure_model(variant: str) -> str:
    """Download selected pose model if not present and return local path.

    Falls back from heavy -> full -> lite if needed.
    """
    preferred = variant if variant in MODEL_VARIANTS else "heavy"
    fallback_order = [preferred] + [v for v in ("full", "lite") if v != preferred]

    for current_variant in fallback_order:
        model_name = MODEL_VARIANTS[current_variant]
        model_path = os.path.join(MODEL_DIR, model_name)
        temp_path = f"{model_path}.download"
        model_url = _model_url(current_variant, model_name)

        if _is_valid_model_file(model_path):
            return model_path
        if os.path.exists(model_path):
            print(f"Aviso: modelo local corrupto en {model_path}. Re-descargando...")
            try:
                os.remove(model_path)
            except OSError:
                pass

        print(f"Descargando modelo de pose ({current_variant}) desde {model_url}...")
        try:
            urllib.request.urlretrieve(model_url, temp_path)
            if not _is_valid_model_file(temp_path):
                raise RuntimeError("archivo descargado invalido o incompleto")
            os.replace(temp_path, model_path)
            print(f"✓ Modelo guardado en {model_path}")
            return model_path
        except urllib.error.HTTPError as e:
            print(f"Aviso: modelo '{current_variant}' no disponible ({e.code}). Probando fallback...")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            continue
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise RuntimeError(f"Fallo al descargar modelo '{current_variant}': {e}")

    raise RuntimeError("No se pudo descargar ningun modelo de pose (heavy/full/lite).")


class PoseDetector:
    def __init__(self, settings: AppSettings):
        self.settings = settings

        # Ensure model is available.
        model_path = ensure_model(settings.pose_model_variant)

        # Use MediaPipe Tasks Vision Pose Landmarker.
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=settings.min_detection_confidence,
            min_pose_presence_confidence=settings.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """Process frame and return landmarks wrapped in a compatible object."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self._landmarker.detect(mp_image)

        # Wrap results in a mock object compatible with skeleton_processor.
        return PoseResults(results.pose_landmarks)

    def close(self) -> None:
        pass


class PoseResults:
    """Mock wrapper for compatibility with skeleton_processor expectations."""

    def __init__(self, landmarks_list):
        if landmarks_list:
            self.pose_landmarks = MockLandmarks(landmarks_list[0])
        else:
            self.pose_landmarks = None


class MockLandmarks:
    """Mock wrapper for a single pose's landmarks."""

    def __init__(self, landmarks):
        self.landmark = landmarks
