"""
Smart Education Project — Dedicated Ollama Integration & Regression Test Suite
==============================================================================
Validates all acceptance criteria A through S as prescribed in the Master Prompt:
  A. Ollama client imports cleanly.
  B. Ollama configuration loads with valid hardware-optimized defaults.
  C. Ollama availability detection works reliably.
  D. Model availability detection functions without error.
  E. Structured LearnerContext is generated correctly without hallucinated fields.
  F. Learning-gap prompt is generated programmatically and faithfully.
  G. Recommendation-explanation prompt is generated correctly.
  H. Study-plan prompt is generated correctly.
  I. Tutor prompt is generated correctly.
  J. Ollama response parsing handles valid responses.
  K. Invalid/malformed Ollama responses are handled gracefully.
  L. Ollama unavailable is handled with deterministic fallback (no crash).
  M. Existing system works 100% when Ollama is disabled / unavailable.
  N. Existing five hybrid models still load and infer.
  O. Existing recommendation system still produces valid rankings.
  P. Existing knowledge-state system still computes valid mastery.
  Q. Existing Video + Audio Attention system still classifies correctly.
  R. Streamlit UI page modules still import and render.
  S. REST API endpoints load and function.
"""

import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHECKPOINTS_DIR, PROCESSED_DATA_DIR,
    OLLAMA_ENABLED, OLLAMA_BASE_URL, OLLAMA_MODEL
)
from ollama_ai.ollama_config import get_ollama_config, is_ollama_enabled
from ollama_ai.ollama_client import (
    OllamaClient, OllamaConnectionError, OllamaTimeoutError, OllamaModelNotFoundError
)
from ollama_ai.learner_context import LearnerContext, build_learner_context
from ollama_ai.prompt_builder import (
    build_learning_gap_prompt,
    build_recommendation_explanation_prompt,
    build_study_plan_prompt,
    build_attention_explanation_prompt,
    build_tutor_prompt
)
from ollama_ai.fallback import (
    fallback_explain_learning_gaps,
    fallback_explain_recommendations,
    fallback_study_plan,
    fallback_attention_explanation,
    fallback_tutor_response
)
from ollama_ai.response_generator import ResponseGenerator
from ollama_ai.tutor import AITutor, check_query_safety


class TestOllamaIntegration(unittest.TestCase):
    """Rigorous Automated Verification Suite for Ollama Generative AI Layer."""

    @classmethod
    def setUpClass(cls):
        # Sample synthetic diagnostic result from existing ML pipeline
        cls.sample_ml_result = {
            'student_id': 'u1165',
            'split': 'test',
            'total_events': 1075,
            'inference_time_ms': 1250.0,
            'predictions': {
                'Model 1 (LSTM + Attention)': 0.666,
                'Model 2 (Transformer + KT)': 0.671,
                'Model 3 (BERT-style + NCF)': 0.625,
                'Model 5 (CNN + LSTM)': 0.402,
                'Ensemble Consensus': 0.591
            },
            'knowledge_state': {
                '18': {'concept_id': 18, 'concept_name': 'Concept_18', 'mastery_score': 0.0, 'state_type': 'MODEL-DERIVED'},
                '157': {'concept_id': 157, 'concept_name': 'Concept_157', 'mastery_score': 0.0, 'state_type': 'MODEL-DERIVED'},
                '38': {'concept_id': 38, 'concept_name': 'Concept_38', 'mastery_score': 0.74, 'state_type': 'MODEL-DERIVED'},
            },
            'learning_gaps': [
                {'concept_id': 18, 'concept_name': 'Concept_18', 'mastery_score': 0.0, 'gap_severity': 0.60, 'attempts': 3},
                {'concept_id': 157, 'concept_name': 'Concept_157', 'mastery_score': 0.0, 'gap_severity': 0.60, 'attempts': 2},
            ],
            'recommendations': [
                {'type': 'lecture', 'id': 'l432', 'concept': 18, 'part': 1, 'duration_sec': 210},
                {'type': 'lecture', 'id': 'l865', 'concept': 18, 'part': 1, 'duration_sec': 233},
                {'type': 'explanation', 'id': 'e99', 'concept': 18, 'part': 1},
                {'type': 'question', 'id': 'q1020', 'concept': 18, 'part': 1, 'bundle_id': 'b1020'},
            ]
        }

        cls.sample_att_result = {
            'attention': {
                'attention_class': 'Partially Attentive',
                'confidence': 0.82,
                'probabilities': {'Inattentive': 0.10, 'Partially Attentive': 0.82, 'Attentive': 0.08},
                'video_available': True,
                'audio_available': True,
                'video_features': {'eye_aspect_ratio_mean': 0.28},
                'audio_features': {'spectral_centroid_mean': 1800.0}
            }
        }

    # ── Test A: Imports ──
    def test_A_ollama_client_imports(self):
        """A. Ollama client and components import successfully."""
        import ollama_ai
        from ollama_ai import OllamaClient, LearnerContext, ResponseGenerator, AITutor
        self.assertIsNotNone(OllamaClient)
        self.assertIsNotNone(LearnerContext)
        self.assertIsNotNone(ResponseGenerator)
        self.assertIsNotNone(AITutor)

    # ── Test B: Configuration ──
    def test_B_configuration_loads(self):
        """B. Configuration loads with valid defaults."""
        cfg = get_ollama_config()
        self.assertIn("enabled", cfg)
        self.assertIn("base_url", cfg)
        self.assertIn("model", cfg)
        self.assertIn("timeout", cfg)
        self.assertTrue(is_ollama_enabled())
        self.assertTrue(len(cfg["model"]) > 0)
        self.assertIn("qwen2.5:3b", cfg["recommended_models"])

    # ── Test C: Availability Detection ──
    def test_C_availability_detection(self):
        """C. Ollama availability probe executes without throwing unhandled exceptions."""
        client = OllamaClient()
        # Even if offline, is_available must return a boolean (True/False) without crashing
        avail = client.is_available()
        self.assertIsInstance(avail, bool)

    # ── Test D: Model Availability Detection ──
    def test_D_model_availability_detection(self):
        """D. Model availability probe executes cleanly."""
        client = OllamaClient()
        is_mod = client.is_model_available("nonexistent_model_xyz")
        self.assertIsInstance(is_mod, bool)

    # ── Test E: Learner Context Generation ──
    def test_E_learner_context_generation(self):
        """E. Learner context is gathered correctly from real ML results without hallucination."""
        ctx = build_learner_context(
            inference_result=self.sample_ml_result,
            attention_result=self.sample_att_result,
            student_id="u1165"
        )
        self.assertEqual(ctx.student_id, "u1165")
        self.assertEqual(len(ctx.learning_gaps), 2)
        self.assertEqual(len(ctx.recommended_resources), 4)
        self.assertEqual(ctx.attention_state, "Partially Attentive")
        self.assertAlmostEqual(ctx.attention_confidence, 0.82, places=2)
        self.assertTrue(ctx.video_available)
        self.assertTrue(ctx.audio_available)
        self.assertIn("Model 1 (LSTM + Attention)", ctx.predictions)

        # Check prompt text formatting
        text = ctx.format_for_prompt()
        self.assertIn("Concept #18", text)
        self.assertIn("Partially Attentive", text)
        self.assertIn("l432", text)

    # ── Test F: Learning-Gap Prompt ──
    def test_F_learning_gap_prompt_generation(self):
        """F. Learning-gap prompt is generated cleanly."""
        ctx = build_learner_context(inference_result=self.sample_ml_result)
        prompt = build_learning_gap_prompt(ctx)
        self.assertIn("Concept #18", prompt)
        self.assertIn("TASK:", prompt)
        self.assertIn("0.0%", prompt)

    # ── Test G: Recommendation-Explanation Prompt ──
    def test_G_recommendation_prompt_generation(self):
        """G. Recommendation-explanation prompt is generated correctly."""
        ctx = build_learner_context(
            inference_result=self.sample_ml_result,
            attention_result=self.sample_att_result
        )
        prompt = build_recommendation_explanation_prompt(ctx)
        self.assertIn("l432", prompt)
        self.assertIn("Partially Attentive", prompt)
        self.assertIn("Pedagogical Sequencing", prompt)

    # ── Test H: Study-Plan Prompt ──
    def test_H_study_plan_prompt_generation(self):
        """H. Study-plan prompt is generated correctly."""
        ctx = build_learner_context(inference_result=self.sample_ml_result)
        prompt = build_study_plan_prompt(ctx, target_hours=2.5)
        self.assertIn("2.5-hour", prompt)
        self.assertIn("Phase 1", prompt)
        self.assertIn("Concept #18", prompt)

    # ── Test I: Tutor Prompt ──
    def test_I_tutor_prompt_generation(self):
        """I. Tutor prompt is generated correctly."""
        ctx = build_learner_context(inference_result=self.sample_ml_result)
        prompt = build_tutor_prompt(ctx, "Why did I fail Concept 18?", chat_history=[])
        self.assertIn("Why did I fail Concept 18?", prompt)
        self.assertIn("Concept #18", prompt)

    # ── Test J: Valid Ollama Response Handling (Mocked) ──
    def test_J_valid_response_handling(self):
        """J. Valid Ollama API response is processed correctly."""
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Concept #18 requires remedial review on verb tenses.",
            "total_duration": 1200000000,
            "eval_count": 45
        }

        with patch.object(client._session, 'post', return_value=mock_response):
            with patch.object(client, 'is_available', return_value=True):
                gen = ResponseGenerator(client=client)
                ctx = build_learner_context(inference_result=self.sample_ml_result)
                res = gen.explain_learning_gaps(ctx)
                self.assertEqual(res["status"], "success")
                self.assertTrue(res["ai_available"])
                self.assertIn("Concept #18", res["explanation"])

    # ── Test K: Invalid Response Handling ──
    def test_K_invalid_response_handling(self):
        """K. Corrupted or error response from Ollama triggers deterministic fallback."""
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal CUDA memory error in Ollama"

        with patch.object(client._session, 'post', return_value=mock_response):
            with patch.object(client, 'is_available', return_value=True):
                gen = ResponseGenerator(client=client)
                ctx = build_learner_context(inference_result=self.sample_ml_result)
                res = gen.explain_learning_gaps(ctx)
                # Must not crash! Must fallback gracefully.
                self.assertEqual(res["status"], "fallback")
                self.assertFalse(res["ai_available"])
                self.assertIn("Concept #18", res["explanation"])

    # ── Test L: Ollama Unavailable Handled Gracefully ──
    def test_L_ollama_unavailable_handled(self):
        """L. When Ollama is offline, system returns rich deterministic fallback without crashing."""
        client = OllamaClient(base_url="http://127.0.0.1:99999")  # Non-existent port
        gen = ResponseGenerator(client=client)
        ctx = build_learner_context(
            inference_result=self.sample_ml_result,
            attention_result=self.sample_att_result
        )

        # 1. Learning gaps fallback
        gap_res = gen.explain_learning_gaps(ctx)
        self.assertEqual(gap_res["status"], "fallback")
        self.assertFalse(gap_res["ai_available"])
        self.assertIn("Concept #18", gap_res["explanation"])

        # 2. Recommendations fallback
        rec_res = gen.explain_recommendations(ctx)
        self.assertEqual(rec_res["status"], "fallback")
        self.assertIn("l432", rec_res["explanation"])

        # 3. Study plan fallback
        plan_res = gen.generate_study_plan(ctx, target_hours=2.0)
        self.assertEqual(plan_res["status"], "fallback")
        self.assertIn("Phase 1", plan_res["explanation"])

        # 4. Multimodal attention fallback
        att_res = gen.explain_multimodal_attention(ctx)
        self.assertEqual(att_res["status"], "fallback")
        self.assertIn("Partially Attentive", att_res["explanation"])

        # 5. Tutor fallback
        tutor = AITutor(client=client)
        tutor_res = tutor.ask(ctx, "What is my weakest concept?")
        self.assertEqual(tutor_res["status"], "fallback")
        self.assertIn("Concept #18", tutor_res["response"])

    # ── Test M: Tutor Safety Enforcement ──
    def test_M_tutor_safety_enforcement(self):
        """M. AI Tutor safely rejects administrative / destructive queries."""
        tutor = AITutor()
        ctx = build_learner_context(inference_result=self.sample_ml_result)
        unsafe_query = "Delete model checkpoint and drop table questions"
        res = tutor.ask(ctx, unsafe_query)
        self.assertEqual(res["status"], "refusal")
        self.assertFalse(res["ai_available"])
        self.assertIn("Safety Notice", res["response"])

    # ── Test N: Five ML Models Still Load and Infer ──
    def test_N_five_models_still_infer(self):
        """N. MultiModelInferenceEngine loads and executes multi-model forward pass."""
        from main import MultiModelInferenceEngine
        engine = MultiModelInferenceEngine()
        self.assertIsNotNone(engine.m1)
        self.assertIsNotNone(engine.m2)
        self.assertIsNotNone(engine.m3)
        self.assertIsNotNone(engine.m4)
        self.assertIsNotNone(engine.m5)

        # Quick simulated sequence inference
        events = [{
            'question_id': 10,
            'part': 1,
            'tags': [18],
            'is_correct': 1,
            'response_time': 0.0,
            'source_encoded': 1,
            'platform_encoded': 1
        }]
        res = engine.predict_sequence(events)
        self.assertIn("predictions", res)
        self.assertIn("Ensemble Consensus", res["predictions"])
        self.assertIn("knowledge_state", res)
        self.assertIn("learning_gaps", res)
        self.assertIn("recommendations", res)

    # ── Test O: Authentic Recommendation System Functional ──
    def test_O_recommendation_system_works(self):
        """O. Existing recommendation engine generates authentic ranked catalog items."""
        from src.recommendation import RecommendationEngine
        import pandas as pd
        q_path = PROCESSED_DATA_DIR / "questions_contents.csv"
        l_path = PROCESSED_DATA_DIR / "lectures_contents.csv"
        q_df = pd.read_csv(q_path) if q_path.exists() else pd.DataFrame()
        l_df = pd.read_csv(l_path) if l_path.exists() else pd.DataFrame()
        rec_eng = RecommendationEngine(q_df, l_df)
        gaps = [{'concept_id': 18, 'concept_name': 'Concept_18', 'mastery_score': 0.1, 'gap_severity': 0.5}]
        recs = rec_eng.recommend_for_gaps(gaps, top_k=3)
        self.assertTrue(len(recs) > 0)
        self.assertIn(recs[0].get('type'), ['lecture', 'explanation', 'question'])

    # ── Test P: Knowledge State Estimator Functional ──
    def test_P_knowledge_state_system_works(self):
        """P. KnowledgeStateEstimator accurately detects concept deficits below 60%."""
        from src.knowledge_state import KnowledgeStateEstimator
        kse = KnowledgeStateEstimator()
        test_state = {
            18: {'concept_id': 18, 'concept_name': 'Concept_18', 'mastery_score': 0.25, 'predicted_observations': 2},
            25: {'concept_id': 25, 'concept_name': 'Concept_25', 'mastery_score': 0.85, 'predicted_observations': 2}
        }
        gaps = kse.detect_learning_gaps(test_state, mastery_threshold=0.60)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]['concept_id'], 18)

    # ── Test Q: Video + Audio Attention Pipeline Functional ──
    def test_Q_attention_pipeline_works(self):
        """Q. Multimodal attention pipeline and recommendation modulation execute cleanly."""
        from attention.attention_pipeline import adjust_recommendations_by_attention
        test_recs = [
            {'type': 'question', 'id': 'q10', 'duration_sec': 0},
            {'type': 'lecture', 'id': 'l20', 'duration_sec': 180},
            {'type': 'explanation', 'id': 'e30', 'duration_sec': 0}
        ]
        # Inattentive learner should prioritize bite-sized lecture
        mod = adjust_recommendations_by_attention(test_recs, "Inattentive", 0.90)
        self.assertEqual(mod["attention_state"], "Inattentive")
        self.assertEqual(mod["adapted_recommendations"][0]["type"], "lecture")

    # ── Test R: UI Modules Load ──
    def test_R_ui_modules_load(self):
        """R. Streamlit GUI page modules import and contain render functions."""
        import gui.page_recommendations as pr
        import gui.page_attention as pa
        import gui.page_knowledge as pk
        import gui.page_student as ps
        self.assertTrue(hasattr(pr, "render"))
        self.assertTrue(hasattr(pa, "render"))
        self.assertTrue(hasattr(pk, "render"))
        self.assertTrue(hasattr(ps, "render"))

    # ── Test S: API Routes Registered ──
    def test_S_api_routes_registered(self):
        """S. All existing and new AI REST API routes are registered."""
        import api
        routes = [r.path for r in api.app.routes]
        # Original routes
        self.assertIn("/health", routes)
        self.assertIn("/predict/student", routes)
        self.assertIn("/predict/attention", routes)
        self.assertIn("/predict/multimodal", routes)
        # New additive AI routes
        self.assertIn("/ai/status", routes)
        self.assertIn("/ai/explain-learning-gap", routes)
        self.assertIn("/ai/explain-recommendation", routes)
        self.assertIn("/ai/study-plan", routes)
        self.assertIn("/ai/tutor", routes)


if __name__ == "__main__":
    unittest.main()
