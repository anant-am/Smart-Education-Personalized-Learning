"""
Smart Education Project — Video + Audio Attention Pipeline CLI Runner
======================================================================
Dedicated runner script for:
1. Training the Multimodal Attention Model (`--train`)
2. Running inference on an actual video file (`--video <path> [--student <id>]`)
3. Automated end-to-end demonstration (`--demo`)
"""

import sys
import os
import argparse
import tempfile
from pathlib import Path
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attention.attention_pipeline import run_video_attention_pipeline
from attention.attention_inference import AttentionInferenceEngine


def create_synthetic_demo_video(duration_sec: int = 4, fps: int = 25, include_audio: bool = True) -> str:
    """Creates a temporary synthetic video with a rendered face and optional audio for testing."""
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required to render synthetic demonstration videos. "
            "Please install opencv-python or opencv-python-headless."
        )
    temp_dir = tempfile.gettempdir()
    silent_path = os.path.join(temp_dir, "smart_education_attention_demo_silent.mp4")
    final_path = os.path.join(temp_dir, "smart_education_attention_demo.mp4")

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(silent_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    for i in range(total_frames):
        # Create dark blue background
        frame = np.full((height, width, 3), (40, 30, 20), dtype=np.uint8)

        # Draw a simulated student face
        center = (width // 2, height // 2)
        cv2.circle(frame, center, 100, (180, 200, 230), -1)  # Skin tone

        # Eyes (blinking every 35 frames)
        is_blink = (i % 35) in [0, 1]
        eye_y = height // 2 - 25
        if not is_blink:
            cv2.circle(frame, (center[0] - 35, eye_y), 15, (255, 255, 255), -1)
            cv2.circle(frame, (center[0] + 35, eye_y), 15, (255, 255, 255), -1)
            cv2.circle(frame, (center[0] - 35, eye_y), 6, (60, 40, 20), -1)
            cv2.circle(frame, (center[0] + 35, eye_y), 6, (60, 40, 20), -1)
        else:
            cv2.line(frame, (center[0] - 50, eye_y), (center[0] - 20, eye_y), (60, 40, 20), 3)
            cv2.line(frame, (center[0] + 20, eye_y), (center[0] + 50, eye_y), (60, 40, 20), 3)

        # Mouth
        cv2.ellipse(frame, (center[0], height // 2 + 40), (30, 15), 0, 0, 180, (80, 80, 180), 3)

        # Overlay text
        cv2.putText(frame, "Smart Education Attention Demo", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(frame)

    out.release()

    # Mux synthetic speech audio track if audio libraries are available
    if include_audio:
        try:
            import soundfile as sf
            import imageio_ffmpeg
            import subprocess

            sample_rate = 16000
            t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
            harmonics = 0.5 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1500 * t) + 0.1 * np.sin(2 * np.pi * 2500 * t)
            envelope = np.clip(np.sin(2 * np.pi * 3 * t), 0, 1) ** 2
            audio_signal = (harmonics * envelope * 0.4).astype(np.float32)

            wav_temp = os.path.join(temp_dir, "temp_demo_speech.wav")
            sf.write(wav_temp, audio_signal, sample_rate)

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-i", silent_path,
                "-i", wav_temp,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                final_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            if os.path.exists(wav_temp):
                os.remove(wav_temp)
            if os.path.exists(silent_path):
                os.remove(silent_path)
            return final_path
        except Exception:
            # Fallback to silent video if muxing fails
            if os.path.exists(silent_path):
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(silent_path, final_path)
            return final_path

    if os.path.exists(silent_path):
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(silent_path, final_path)
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Smart Education — Video + Audio Attention Subsystem")
    parser.add_argument("--train", action="store_true", help="Execute complete 10-fold CV training and evaluation lifecycle")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (.mp4/.avi/.mov/.mkv)")
    parser.add_argument("--student", type=str, default=None, help="Optional student ID to integrate (e.g., u1165)")
    parser.add_argument("--demo", action="store_true", help="Execute automated demo with synthetic video and student u1165")
    args = parser.parse_args()

    if args.train:
        from attention.attention_evaluator import AttentionEvaluator
        evaluator = AttentionEvaluator()
        evaluator.run_full_lifecycle()
        return

    if args.demo:
        print("\n" + "=" * 80)
        print("  SMART EDUCATION — VIDEO + AUDIO ATTENTION DEMONSTRATION")
        print("=" * 80)
        demo_video = create_synthetic_demo_video(duration_sec=3)
        print(f"  Generated test video: {demo_video}")
        print("  Running integrated pipeline for Student 'u1165'...")

        res = run_video_attention_pipeline(video_path=demo_video, student_id="u1165")

        att = res["attention"]
        print("\n[1] ATTENTION DETECTION RESULTS:")
        print(f"  Status:             {att.get('status')}")
        print(f"  Predicted Class:    {att.get('attention_class')}")
        print(f"  Confidence:         {att.get('confidence', 0.0)*100:.1f}%")
        print(f"  Video Modality:     {'Active' if att.get('video_available') else 'Unavailable'}")
        print(f"  Audio Modality:     {'Active' if att.get('audio_available') else 'Unavailable'}")
        print("  Probability Distribution:")
        for k, v in att.get("probabilities", {}).items():
            bar = "=" * int(v * 25)
            print(f"    - {k:<20}: {v*100:5.1f}% [{bar}]")

        pers = res.get("personalization")
        if pers:
            print("\n[2] ATTENTION-ADAPTED RECOMMENDATION PERSONALIZATION:")
            print(f"  Strategy: {pers.get('strategy_title')}")
            print(f"  Rationale: {pers.get('rationale')}")
            print("  Top Adapted Recommendations:")
            for i, r in enumerate(pers.get("adapted_recommendations", [])[:4], 1):
                pacing = f"[{r.get('pacing_tag')}]" if "pacing_tag" in r else ""
                print(f"    {i}. {r.get('type').upper():<11} ID: {r.get('id')} | Targets Concept {r.get('concept')} {pacing}")

        print(f"\n  Pipeline Time: {res.get('pipeline_execution_time_ms', 0):.2f} ms")
        print("=" * 80)

        # Cleanup demo video
        try:
            os.remove(demo_video)
        except Exception:
            pass
        return

    if args.video:
        print(f"\n  Executing Attention Pipeline on {args.video}...")
        res = run_video_attention_pipeline(video_path=args.video, student_id=args.student)
        att = res["attention"]
        print(f"  Attention Class: {att.get('attention_class')} (Confidence: {att.get('confidence', 0)*100:.1f}%)")
        print("  Probabilities:", att.get("probabilities"))
        if res.get("personalization"):
            print("  Adapted Strategy:", res["personalization"].get("strategy_title"))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
