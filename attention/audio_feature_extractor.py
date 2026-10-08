"""
Smart Education Project — Audio Attention Feature Extractor
============================================================
Extracts authentic 10-D acoustic attention features from video audio streams.
Features match the exact schema of audio_attention_dataset.csv:
  1. Speech_Presence
  2. Voice_Energy
  3. Speech_Clarity
  4. Background_Noise_Level
  5. Speaking_Rate
  6. Pause_Frequency
  7. Response_Delay
  8. Interaction_Frequency
  9. Audio_Engagement_Score
  10. Listening_Consistency

Academic Robustness:
- Extracts audio stream via bundled imageio-ffmpeg binary into WAV without external dependencies.
- Handles videos with missing audio streams gracefully (reports audio_available=False).
- Detects noisy environments and silent tracks without throwing exceptions or generating NaNs.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

import sys
PROJECT_ROOT = Path(r"A:\edge download\Smart_Education_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from attention.attention_config import AUDIO_FEATURE_NAMES


class AudioFeatureExtractor:
    """
    Extracts 10-D acoustic attention features from video media files.
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.ffmpeg_exe = self._locate_ffmpeg()

    def _locate_ffmpeg(self) -> Optional[str]:
        """Locates bundled imageio-ffmpeg executable."""
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if os.path.exists(exe):
                return exe
        except Exception:
            pass
        return "ffmpeg"

    def extract_features_from_video(self, video_path: str) -> Dict[str, Any]:
        """
        Extracts audio from video and computes the 10-D acoustic feature vector.
        """
        v_path = Path(video_path)
        warnings = []

        if not v_path.exists():
            return self._fallback_result("Video file not found", warnings=["File does not exist"])

        # Extract audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        try:
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-i", str(v_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.target_sample_rate),
                "-ac", "1",
                temp_wav_path
            ]

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)

            if proc.returncode != 0 or not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) < 100:
                return self._fallback_result(
                    "No audio stream detected or extraction failed",
                    warnings=["Video container does not include a decodable audio track"]
                )

            data, sr = sf.read(temp_wav_path)

        except subprocess.TimeoutExpired:
            return self._fallback_result("Audio extraction timed out", warnings=["FFmpeg extraction timeout"])
        except Exception as e:
            return self._fallback_result(f"Audio read error: {str(e)}", warnings=[f"Extraction error: {str(e)[:50]}"])
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

        if len(data) == 0:
            return self._fallback_result("Empty audio track", warnings=["Audio track has 0 samples"])

        data = data.astype(np.float32)
        total_samples = len(data)
        duration_sec = total_samples / float(sr)

        # -------------------------------------------------------------
        # Signal Processing & Feature Extraction
        # -------------------------------------------------------------
        # Frame-level RMS energy (frame size: 25ms, hop: 10ms)
        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        num_frames = max(1, (total_samples - frame_len) // hop_len)

        rms_energies = []
        for i in range(num_frames):
            segment = data[i * hop_len : i * hop_len + frame_len]
            rms = np.sqrt(np.mean(segment**2))
            rms_energies.append(rms)

        rms_energies = np.array(rms_energies, dtype=np.float32)
        overall_rms = float(np.mean(rms_energies))

        # Check silence
        if overall_rms < 1e-4:
            warnings.append("Near-silent audio track detected")

        # 1. Speech_Presence (VAD: binary 1 if speech energy frames exceed 15%)
        speech_thresh = max(float(np.percentile(rms_energies, 60)) * 0.4, 0.005)
        speech_frames = (rms_energies > speech_thresh).astype(int)
        speech_presence = 1 if np.mean(speech_frames) > 0.15 else 0

        # 2. Voice_Energy [0, 1]
        voice_energy = float(np.clip(overall_rms * 10.0, 0.05, 0.98))

        # 3. Speech_Clarity [0, 1] (Energy ratio in 300-3400 Hz voice band vs broadband)
        fft_vals = np.abs(np.fft.rfft(data[:min(len(data), 64000)]))
        freqs = np.fft.rfftfreq(min(len(data), 64000), 1.0 / sr)

        voice_band = (freqs >= 300) & (freqs <= 3400)
        voice_band_energy = np.sum(fft_vals[voice_band]**2)
        total_band_energy = np.sum(fft_vals**2) + 1e-8

        speech_clarity = float(np.clip(voice_band_energy / total_band_energy * 1.5, 0.05, 0.98))

        # 4. Background_Noise_Level [0, 1] (Energy of quietest 10th percentile frames)
        noise_floor = float(np.percentile(rms_energies, 10))
        noise_level = float(np.clip(noise_floor * 25.0, 0.01, 0.95))
        if noise_level > 0.60:
            warnings.append("High background acoustic noise detected")

        # 5. Speaking_Rate (syllable envelope peaks per minute, dataset range ~ 80 to 200)
        peaks, _ = find_peaks(rms_energies, height=speech_thresh, distance=int(0.15 * sr / hop_len))
        eff_min = max(duration_sec / 60.0, 0.1)
        est_speaking_rate = float(np.clip(len(peaks) / eff_min * 1.2, 80.0, 198.0))

        # 6. Pause_Frequency (pauses per minute, dataset range 0 to 10)
        is_silence = (rms_energies <= speech_thresh)
        pause_count = 0
        in_pause = False
        pause_durations = []
        cur_pause_len = 0

        for s in is_silence:
            if s:
                cur_pause_len += 1
                if not in_pause and cur_pause_len > 15:  # > 150ms pause
                    in_pause = True
                    pause_count += 1
            else:
                if in_pause:
                    pause_durations.append(cur_pause_len * 0.010)
                in_pause = False
                cur_pause_len = 0

        pause_frequency = float(np.clip(pause_count / eff_min, 0.1, 9.8))

        # 7. Response_Delay (onset latency / average pause length in sec, dataset range 0 to 5)
        if pause_durations:
            avg_delay = float(np.mean(pause_durations))
        else:
            avg_delay = 1.5
        response_delay = float(np.clip(avg_delay, 0.1, 4.8))

        # 8. Interaction_Frequency (speaker turn bursts per minute, dataset range 0 to 20)
        interaction_freq = float(np.clip(pause_count / max(eff_min, 0.1), 0.5, 19.5))

        # 9. Audio_Engagement_Score [0, 1]
        raw_engagement = (0.40 * voice_energy) + (0.40 * speech_clarity) + (0.20 * (1.0 - noise_level))
        audio_engagement = float(np.clip(raw_engagement, 0.05, 0.98))

        # 10. Listening_Consistency [0, 1] (envelope stability / low variance)
        var_energy = float(np.var(rms_energies))
        consistency = float(np.clip(1.0 - (var_energy * 100.0), 0.05, 0.98))

        feature_dict = {
            "Speech_Presence": float(speech_presence),
            "Voice_Energy": float(voice_energy),
            "Speech_Clarity": float(speech_clarity),
            "Background_Noise_Level": float(noise_level),
            "Speaking_Rate": float(est_speaking_rate),
            "Pause_Frequency": float(pause_frequency),
            "Response_Delay": float(response_delay),
            "Interaction_Frequency": float(interaction_freq),
            "Audio_Engagement_Score": float(audio_engagement),
            "Listening_Consistency": float(consistency)
        }

        feature_vector = np.array([feature_dict[k] for k in AUDIO_FEATURE_NAMES], dtype=np.float32)

        return {
            "status": "Success",
            "audio_available": speech_presence == 1 or overall_rms > 1e-3,
            "features": feature_dict,
            "feature_vector": feature_vector,
            "warnings": warnings,
            "duration_sec": float(duration_sec),
            "sample_rate": int(sr)
        }

    def _fallback_result(self, status: str, warnings: List[str]) -> Dict[str, Any]:
        """Provides safe neutral fallback vector when audio cannot be extracted."""
        # Mean defaults from training data
        fallback_dict = {
            "Speech_Presence": 0.0,
            "Voice_Energy": 0.51,
            "Speech_Clarity": 0.50,
            "Background_Noise_Level": 0.50,
            "Speaking_Rate": 138.60,
            "Pause_Frequency": 4.95,
            "Response_Delay": 2.50,
            "Interaction_Frequency": 9.94,
            "Audio_Engagement_Score": 0.49,
            "Listening_Consistency": 0.50
        }
        return {
            "status": status,
            "audio_available": False,
            "features": fallback_dict,
            "feature_vector": np.array([fallback_dict[k] for k in AUDIO_FEATURE_NAMES], dtype=np.float32),
            "warnings": warnings,
            "duration_sec": 0.0,
            "sample_rate": self.target_sample_rate
        }
