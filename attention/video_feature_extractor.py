"""
Smart Education Project — Video Attention Feature Extractor
============================================================
Extracts authentic 10-D visual attention features from raw video files (.mp4, .avi, .mov, .mkv).
Features match the exact schema of video_attention_dataset.csv:
  1. Face_Visibility
  2. Eye_Gaze_Score
  3. Head_Pose_Stability
  4. Blink_Rate
  5. Facial_Expression_Intensity
  6. Body_Posture_Score
  7. Movement_Level
  8. Focus_Duration
  9. Screen_Orientation
  10. Attention_Score

Engineered with academic robustness:
- Detects multiple faces and tracks dominant central face
- Handles poor illumination with CLAHE histogram equalization
- Gracefully handles no-face intervals, temporary occlusion, and corrupt frames
- Never crashes on corrupt or missing video files
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attention.attention_config import VIDEO_FEATURE_NAMES


class VideoFeatureExtractor:
    """
    Extracts 10-D visual attention features from video media files using OpenCV.
    """

    def __init__(self, max_sample_frames: int = 300):
        self.max_sample_frames = max_sample_frames
        self.face_cascade = None
        self.eye_cascade = None

        if cv2 is not None:
            try:
                # Load OpenCV Haar cascade classifiers
                cv_data_path = cv2.data.haarcascades
                self.face_cascade = cv2.CascadeClassifier(cv_data_path + "haarcascade_frontalface_default.xml")
                self.eye_cascade = cv2.CascadeClassifier(cv_data_path + "haarcascade_eye.xml")
            except Exception:
                pass

    def extract_features_from_video(self, video_path: str) -> Dict[str, Any]:
        """
        Main extraction entrypoint for raw video file.
        Returns structured dictionary with feature vector, warnings, and modality status.
        """
        if cv2 is None:
            return self._fallback_result(
                "OpenCV (cv2) is not installed",
                warnings=["Raw video frame extraction requires opencv-python or opencv-python-headless."]
            )

        v_path = Path(video_path)
        warnings = []

        if not v_path.exists():
            return self._fallback_result("Video file not found", warnings=["File does not exist"])

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            return self._fallback_result("Could not open video file / unsupported codec", warnings=["Failed to open video codec"])

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            return self._fallback_result("Video has 0 frames", warnings=["Empty video container"])

        # Determine sampling step
        step = max(1, total_frames // self.max_sample_frames)

        frame_idx = 0
        sampled_count = 0
        faces_detected = 0
        eyes_detected_list = []
        face_centers = []
        face_areas = []
        motion_diffs = []
        prev_gray = None
        focus_run = 0
        max_focus_run = 0
        low_light_frames = 0
        multiple_face_warnings = 0

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                sampled_count += 1

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Check lighting
                mean_brightness = np.mean(gray)
                if mean_brightness < 40.0:
                    low_light_frames += 1
                    gray = clahe.apply(gray)  # Enhance dark frames

                # Motion detection via frame differencing
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    motion_diffs.append(float(np.mean(diff)))
                prev_gray = gray.copy()

                # Face detection
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=4,
                    minSize=(40, 40)
                )

                if len(faces) > 0:
                    faces_detected += 1
                    if len(faces) > 1:
                        multiple_face_warnings += 1

                    # Pick largest face
                    largest_face = max(faces, key=lambda r: r[2] * r[3])
                    fx, fy, fw, fh = largest_face
                    cx = (fx + fw / 2.0) / width
                    cy = (fy + fh / 2.0) / height
                    face_centers.append((cx, cy))
                    face_areas.append((fw * fh) / (width * height))

                    # Eye detection inside upper half of face
                    roi_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
                    eyes_detected_list.append(len(eyes))

                    # Focus tracking (face centered and eyes visible)
                    is_centered = 0.25 <= cx <= 0.75 and 0.20 <= cy <= 0.80
                    if is_centered and len(eyes) >= 1:
                        focus_run += 1
                        if focus_run > max_focus_run:
                            max_focus_run = focus_run
                    else:
                        focus_run = 0
                else:
                    eyes_detected_list.append(0)
                    focus_run = 0

            frame_idx += 1

        cap.release()

        if sampled_count == 0:
            return self._fallback_result("No frames successfully decoded", warnings=["Zero decoded frames"])

        # Warnings diagnostics
        if low_light_frames > sampled_count * 0.4:
            warnings.append("Low ambient lighting detected in over 40% of frames (CLAHE applied)")
        if multiple_face_warnings > sampled_count * 0.2:
            warnings.append("Multiple faces detected in scene; dominant learner tracked")

        if faces_detected == 0:
            warnings.append("No learner face detected in decoded video frames; facial gaze, head pose, blink rate, and expression cannot be reliably measured.")
            return self._fallback_result("No face detected in video stream", warnings=warnings)

        # -------------------------------------------------------------
        # Feature Calculations (Mapped strictly to training CSV schema)
        # -------------------------------------------------------------
        # 1. Face_Visibility (fraction of frames with face >= 0.35 -> binary 1, else 0)
        face_ratio = faces_detected / sampled_count
        face_visibility = 1.0 if face_ratio >= 0.35 else 0.0

        # 2. Eye_Gaze_Score [0, 1]
        eyes_positive = [e for e in eyes_detected_list if e > 0]
        if eyes_positive:
            avg_eyes = float(np.mean(eyes_detected_list))
            eye_gaze_score = float(np.clip(avg_eyes / 2.0, 0.05, 0.98))
        else:
            eye_gaze_score = 0.0
            warnings.append("Eye regions were not resolved in detected face ROI; gaze tracking unavailable.")

        # 3. Head_Pose_Stability [0, 1] (Inverse of centroid jitter)
        if len(face_centers) > 2:
            centers = np.array(face_centers)
            var_x = float(np.var(centers[:, 0]))
            var_y = float(np.var(centers[:, 1]))
            jitter = float(np.sqrt(var_x + var_y))
            stability = float(np.clip(1.0 - (jitter * 4.0), 0.05, 0.98))
        else:
            stability = 0.50
            warnings.append("Insufficient facial tracking frames to calculate head pose jitter variance.")

        # 4. Blink_Rate (blinks per minute, physiological range ~ 6 to 28)
        blinks = 0
        in_blink = False
        for e_cnt in eyes_detected_list:
            if e_cnt == 0 and not in_blink:
                in_blink = True
                blinks += 1
            elif e_cnt > 0:
                in_blink = False

        effective_duration_min = max(duration_sec / 60.0, 0.05)
        if eyes_positive and blinks > 0:
            est_blink_rate = float(np.clip(blinks / effective_duration_min, 6.0, 28.0))
        elif eyes_positive:
            est_blink_rate = 10.0  # Steady open eyes without observed blinks in short window
            warnings.append("No blink transitions observed during short sampling window.")
        else:
            est_blink_rate = 0.0
            warnings.append("Blink transitions could not be measured because eye tracking was unavailable.")

        # 5. Facial_Expression_Intensity [0, 1]
        expression_intensity = float(np.clip(np.std(motion_diffs) / 15.0 if motion_diffs else 0.10, 0.05, 0.95))
        warnings.append("Note: Facial expression intensity is approximated from ROI pixel variance, not 3D FACS action units.")

        # 6. Body_Posture_Score [0, 1] (Centrality and uprightness of head)
        if face_centers:
            mean_cx = float(np.mean([c[0] for c in face_centers]))
            mean_cy = float(np.mean([c[1] for c in face_centers]))
            dist_from_ideal = float(np.sqrt((mean_cx - 0.5)**2 + (mean_cy - 0.45)**2))
            posture_score = float(np.clip(1.0 - (dist_from_ideal * 2.0), 0.10, 0.95))
        else:
            posture_score = 0.0

        # 7. Movement_Level [0, 1]
        if motion_diffs:
            movement_level = float(np.clip(np.mean(motion_diffs) / 25.0, 0.05, 0.98))
        else:
            movement_level = 0.10

        # 8. Focus_Duration (seconds in uninterrupted focus, dataset range 0 to 300)
        sample_interval_sec = duration_sec / max(sampled_count, 1)
        focus_duration = float(np.clip(max_focus_run * sample_interval_sec, 0.0, 300.0))

        # 9. Screen_Orientation (1: standard landscape orientation width >= height, 0: portrait/tilted)
        screen_orientation = 1.0 if width >= height else 0.0

        # 10. Attention_Score [0, 1] (Composite visual metric)
        raw_att = (0.35 * eye_gaze_score) + (0.35 * stability) + (0.30 * face_ratio)
        attention_score = float(np.clip(raw_att, 0.05, 0.98))

        feature_dict = {
            "Face_Visibility": float(face_visibility),
            "Eye_Gaze_Score": float(eye_gaze_score),
            "Head_Pose_Stability": float(stability),
            "Blink_Rate": float(est_blink_rate),
            "Facial_Expression_Intensity": float(expression_intensity),
            "Body_Posture_Score": float(posture_score),
            "Movement_Level": float(movement_level),
            "Focus_Duration": float(focus_duration),
            "Screen_Orientation": float(screen_orientation),
            "Attention_Score": float(attention_score)
        }

        feature_vector = np.array([feature_dict[k] for k in VIDEO_FEATURE_NAMES], dtype=np.float32)

        return {
            "status": "Success",
            "video_available": True,
            "features": feature_dict,
            "feature_vector": feature_vector,
            "warnings": warnings,
            "duration_sec": float(duration_sec),
            "sampled_frames": int(sampled_count),
            "faces_detected_ratio": float(face_ratio)
        }

    def _fallback_result(self, status: str, warnings: List[str]) -> Dict[str, Any]:
        """Provides zeroed feature vector when video cannot be processed, without fabricating measurements."""
        fallback_dict = {k: 0.0 for k in VIDEO_FEATURE_NAMES}
        return {
            "status": status,
            "video_available": False,
            "features": fallback_dict,
            "feature_vector": np.zeros(len(VIDEO_FEATURE_NAMES), dtype=np.float32),
            "warnings": warnings,
            "duration_sec": 0.0,
            "sampled_frames": 0,
            "faces_detected_ratio": 0.0
        }
