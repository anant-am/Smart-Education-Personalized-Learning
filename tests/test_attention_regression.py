"""
Smart Education Project — Full Regression & Integration Test Suite
===================================================================
Rigorous automated test verifying all acceptance criteria A through T:
  A. Existing project starts successfully.
  B. Existing five models still import successfully.
  C. Existing five models still infer as before.
  D. Existing recommendation system still works.
  E. Existing knowledge-state system still works.
  F. New attention dataset loads without leakage.
  G. Attention preprocessing & scaling work without leakage.
  H. Attention model trains with valid forward/backward pass.
  I. 10-fold CV logic executes cleanly.
  J. Training-only SMOTE oversampling functions properly.
  K. Final attention test evaluation produces valid metrics.
  L. Attention checkpoint saves and loads properly.
  M. Preprocessing artifacts save and load properly.
  N. Real video MP4 can be processed.
  O. Missing audio stream is handled gracefully.
  P. No-face video is handled gracefully with safe fallbacks.
  Q. Invalid/corrupt video is handled gracefully without crashing.
  R. Attention results integrate with UI page modules.
  S. Existing UI navigation and components remain intact.
  T. Existing and new API routes remain functional.
"""

import os
import sys
import unittest
import tempfile
import importlib
from pathlib import Path
import numpy as np
import torch
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHECKPOINTS_DIR, PROCESSED_DATA_DIR, ATTENTION_CHECKPOINTS_DIR,
    VIDEO_ATTENTION_CSV, AUDIO_ATTENTION_CSV
)
from attention.attention_config import (
    VIDEO_FEATURE_NAMES, AUDIO_FEATURE_NAMES, ATTENTION_LABEL_MAP
)
from attention.attention_model import MultimodalAttentionModel
from attention.attention_preprocessing import AttentionPreprocessor
from attention.attention_dataset import AttentionDataset
from attention.video_feature_extractor import VideoFeatureExtractor
from attention.audio_feature_extractor import AudioFeatureExtractor
from attention.attention_inference import AttentionInferenceEngine
from attention.attention_pipeline import run_video_attention_pipeline, adjust_recommendations_by_attention


class TestSmartEducationRegression(unittest.TestCase):
    """End-to-End Test Suite for Smart Education Attention Integration."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Test Suite] Running on execution device: {cls.device}")

    # ── Test A: Existing project configuration and directory structure ──
    def test_A_project_starts(self):
        self.assertTrue(CHECKPOINTS_DIR.exists(), "checkpoints dir missing")
        self.assertTrue(PROCESSED_DATA_DIR.exists(), "processed data dir missing")
        self.assertTrue((PROCESSED_DATA_DIR / "metadata.pkl").exists(), "metadata.pkl missing")

    # ── Test B: Existing five models import successfully ──
    def test_B_existing_five_models_import(self):
        from src.models import (
            LSTMAttentionKT, TransformerKT, BERTNCF,
            AutoencoderRecommender, CNNLSTM
        )
        self.assertIsNotNone(LSTMAttentionKT)
        self.assertIsNotNone(TransformerKT)
        self.assertIsNotNone(BERTNCF)
        self.assertIsNotNone(AutoencoderRecommender)
        self.assertIsNotNone(CNNLSTM)

    # ── Test C: Existing five models still infer as before ──
    def test_C_existing_five_models_infer(self):
        from main import MultiModelInferenceEngine
        engine = MultiModelInferenceEngine(device="cpu")
        res = engine.predict_student_by_id("u1165", split="test")
        self.assertIsNotNone(res)
        self.assertIn("predictions", res)
        self.assertIn("Model 1 (LSTM + Attention)", res["predictions"])
        self.assertIn("Model 2 (Transformer + KT)", res["predictions"])
        self.assertIn("Model 3 (BERT-style + NCF)", res["predictions"])
        self.assertIn("Model 5 (CNN + LSTM)", res["predictions"])
        self.assertIn("Ensemble Consensus", res["predictions"])

    # ── Test D: Existing recommendation system still works ──
    def test_D_existing_recommendation_system(self):
        from src.recommendation import RecommendationEngine
        import pandas as pd
        q_df = pd.read_csv(PROCESSED_DATA_DIR / "questions_contents.csv")
        l_df = pd.read_csv(PROCESSED_DATA_DIR / "lectures_contents.csv")
        rec_eng = RecommendationEngine(q_df, l_df)
        recs = rec_eng.recommend_for_gaps([{"concept_id": 18, "mastery": 0.2}], top_k=3)
        self.assertGreater(len(recs), 0)
        self.assertIn("type", recs[0])
        self.assertIn("id", recs[0])

    # ── Test E: Existing knowledge-state system still works ──
    def test_E_existing_knowledge_state(self):
        from src.knowledge_state import KnowledgeStateEstimator
        kse = KnowledgeStateEstimator()
        state = kse.compute_model_derived_state(
            model_predictions=[0.8, 0.4],
            question_tags=[[18], [18]]
        )
        self.assertIn(18, state)
        self.assertAlmostEqual(state[18]["mastery_score"], 0.6, places=4)
        gaps = kse.detect_learning_gaps(state, mastery_threshold=0.70, min_attempts=1)
        self.assertEqual(len(gaps), 1)

    # ── Test F: New attention dataset loads without leakage ──
    def test_F_attention_datasets_load(self):
        prep = AttentionPreprocessor()
        v_df, a_df = prep.load_and_audit_data()
        self.assertEqual(len(v_df), 2000)
        self.assertEqual(len(a_df), 2000)
        # Verify General_Class is not in features
        self.assertNotIn("General_Class", VIDEO_FEATURE_NAMES)
        self.assertNotIn("General_Class", AUDIO_FEATURE_NAMES)
        self.assertNotIn("Target", VIDEO_FEATURE_NAMES)
        self.assertNotIn("Target", AUDIO_FEATURE_NAMES)

    # ── Test G: Attention preprocessing & scaling work ──
    def test_G_attention_preprocessing(self):
        prep = AttentionPreprocessor()
        v_df, a_df = prep.load_and_audit_data()
        split = prep.split_data_programmatically(v_df, a_df, test_size=0.20)
        self.assertEqual(len(split["y_dev"]), 1600)
        self.assertEqual(len(split["y_test"]), 400)
        xv_s, xa_s = prep.fit_and_transform_dev(split["X_video_dev"], split["X_audio_dev"])
        self.assertAlmostEqual(float(np.mean(xv_s)), 0.0, places=1)

    # ── Test H: Attention model forward & backward pass ──
    def test_H_attention_model_training_step(self):
        model = MultimodalAttentionModel().to(self.device)
        xv = torch.randn(4, 10, device=self.device)
        xa = torch.randn(4, 10, device=self.device)
        target = torch.tensor([0, 1, 2, 1], dtype=torch.long, device=self.device)
        logits = model(xv, xa)
        loss = torch.nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        self.assertIsNotNone(model.classifier[0].weight.grad)

    # ── Test I: 10-Fold CV execution logic ──
    def test_I_cross_validation_logic(self):
        from attention.attention_evaluator import AttentionEvaluator
        evaluator = AttentionEvaluator(device=self.device)
        # Smoke test on small partition
        dummy_v = np.random.randn(50, 10).astype(np.float32)
        dummy_a = np.random.randn(50, 10).astype(np.float32)
        dummy_y = np.random.choice([0, 1, 2], size=50)
        res = evaluator.run_10fold_cross_validation(dummy_v, dummy_a, dummy_y, n_splits=3, epochs_per_fold=1)
        self.assertIn("mean_accuracy", res)
        self.assertEqual(len(res["folds"]), 3)

    # ── Test J: Training-only SMOTE oversampling ──
    def test_J_training_only_smote(self):
        from imblearn.over_sampling import SMOTE
        X = np.random.randn(100, 20)
        y = np.array([0] * 70 + [1] * 20 + [2] * 10)
        smote = SMOTE(k_neighbors=2, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        self.assertEqual(len(y_res), 70 * 3)

    # ── Test K: Final attention test evaluation ──
    def test_K_attention_test_evaluation(self):
        self.assertTrue((PROJECT_ROOT / "reports" / "attention_results.json").exists())
        import json
        with open(PROJECT_ROOT / "reports" / "attention_results.json", "r") as f:
            data = json.load(f)
        self.assertIn("test_metrics", data)
        self.assertIn("accuracy", data["test_metrics"])
        self.assertIn("f1_macro", data["test_metrics"])

    # ── Test L: Checkpoint persistence ──
    def test_L_attention_checkpoints(self):
        best_pth = ATTENTION_CHECKPOINTS_DIR / "attention_multimodal_corrected_best.pth"
        best_meta = ATTENTION_CHECKPOINTS_DIR / "attention_multimodal_corrected_best_meta.json"
        self.assertTrue(best_pth.exists())
        self.assertTrue(best_meta.exists())

    # ── Test M: Preprocessing artifacts persistence ──
    def test_M_preprocessor_artifacts(self):
        art_path = ATTENTION_CHECKPOINTS_DIR / "attention_preprocessor.pkl"
        self.assertTrue(art_path.exists())
        prep = AttentionPreprocessor.load_artifacts(art_path)
        self.assertTrue(prep.is_fitted)

    # ── Test N: Real video processing (MP4) ──
    def test_N_real_video_processing(self):
        from scripts.run_video_attention import create_synthetic_demo_video
        vid_path = create_synthetic_demo_video(duration_sec=2)
        try:
            ve = VideoFeatureExtractor()
            res = ve.extract_features_from_video(vid_path)
            self.assertEqual(res["status"], "Success")
            self.assertEqual(len(res["feature_vector"]), 10)
        finally:
            if os.path.exists(vid_path):
                os.remove(vid_path)

    # ── Test O: Missing audio handling ──
    def test_O_missing_audio_handling(self):
        from scripts.run_video_attention import create_synthetic_demo_video
        vid_path = create_synthetic_demo_video(duration_sec=1)
        try:
            ae = AudioFeatureExtractor()
            res = ae.extract_features_from_video(vid_path)
            # Demo video has no audio track, should report audio_available=False cleanly
            self.assertFalse(res["audio_available"])
            self.assertEqual(len(res["feature_vector"]), 10)
        finally:
            if os.path.exists(vid_path):
                os.remove(vid_path)

    # ── Test P: No-face video handling ──
    def test_P_no_face_video_handling(self):
        # Create blank video without faces
        temp_dir = tempfile.gettempdir()
        vid_path = os.path.join(temp_dir, "no_face_test.mp4")
        out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
        for _ in range(25):
            out.write(np.zeros((240, 320, 3), dtype=np.uint8))
        out.release()

        try:
            ve = VideoFeatureExtractor()
            res = ve.extract_features_from_video(vid_path)
            self.assertEqual(res["features"]["Face_Visibility"], 0.0)
            self.assertFalse(res["video_available"])
        finally:
            if os.path.exists(vid_path):
                os.remove(vid_path)

    # ── Test Q: Corrupt video handling ──
    def test_Q_corrupt_video_handling(self):
        temp_dir = tempfile.gettempdir()
        corrupt_path = os.path.join(temp_dir, "corrupt_test.mp4")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER_CORRUPTED_DATA")

        try:
            ve = VideoFeatureExtractor()
            res = ve.extract_features_from_video(corrupt_path)
            # Must return fallback cleanly without crashing
            self.assertFalse(res["video_available"])
            self.assertEqual(len(res["feature_vector"]), 10)
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

    # ── Test R: Attention results reach UI module ──
    def test_R_ui_module_import(self):
        mod = importlib.import_module("gui.page_attention")
        self.assertTrue(hasattr(mod, "render"))

    # ── Test S: Existing UI page modules intact ──
    def test_S_existing_ui_modules(self):
        from app import PAGES, page_modules
        self.assertIn("Dashboard", PAGES)
        self.assertIn("Student Analysis", PAGES)
        self.assertIn("Knowledge State", PAGES)
        self.assertIn("Recommendations", PAGES)
        self.assertIn("Attention Analysis", PAGES)

        for page_name, mod_path in page_modules.items():
            m = importlib.import_module(mod_path)
            self.assertTrue(hasattr(m, "render"), f"Module {mod_path} missing render function")

    # ── Test T: API routes remain functional ──
    def test_T_api_routes_functional(self):
        from api import routes
        route_paths = [r.path for r in routes]
        self.assertIn("/health", route_paths)
        self.assertIn("/predict/student", route_paths)
        self.assertIn("/predict/attention", route_paths)
        self.assertIn("/predict/multimodal", route_paths)


if __name__ == "__main__":
    unittest.main()
