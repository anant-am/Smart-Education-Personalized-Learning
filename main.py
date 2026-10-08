"""
Smart Education Project — Interactive Multi-Model Master Inference System
==========================================================================
Unified Master Entry Point:
1. Loads all five pre-trained PyTorch deep-learning models simultaneously from
   `models/checkpoints/` directly into memory / NVIDIA GPU without retraining.
2. Accepts interactive student input via terminal:
   - Mode A: Real Student ID lookup from unseen test or dev dataset.
   - Mode B: Custom / simulated student question interaction sequence.
3. Executes simultaneous multi-model forward passes across:
   - Model 1: LSTM + Attention Knowledge Tracing
   - Model 2: Transformer + Knowledge Tracing
   - Model 3: BERT-style Transformer + Neural Collaborative Filtering
   - Model 4: Autoencoder + Recommender Network
   - Model 5: CNN + LSTM Temporal Knowledge Tracing
4. Outputs real-time:
   - Multi-model predictions comparison & ensemble consensus
   - Diagnosed concept mastery (Knowledge State)
   - Pinpointed learning gaps and deficiencies
   - Ranked authentic EdNet remedial recommendations (Lectures > Explanations > Questions)
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR,
    DEFAULT_SEQUENCE_LENGTH, RANDOM_SEED, ensure_dirs
)
from src.models import (
    LSTMAttentionKT, TransformerKT, BERTNCF,
    AutoencoderRecommender, CNNLSTM
)
from src.dataset import KTSequenceDataset
from src.knowledge_state import KnowledgeStateEstimator
from src.recommendation import RecommendationEngine

ensure_dirs()
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class MultiModelInferenceEngine:
    """
    Unified Inference Engine that hosts and orchestrates all 5 trained models.
    """
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print("=" * 80)
        print("  SMART EDUCATION — MULTI-MODEL INFERENCE SYSTEM")
        print("=" * 80)
        print(f"  Execution Device: {self.device}")
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU Hardware:     {gpu_name} ({vram_gb:.1f} GB VRAM)")

        self._load_metadata()
        self._load_recommendation_catalog()
        self._load_models()
        self.kse = KnowledgeStateEstimator()

    def _load_metadata(self):
        meta_path = PROCESSED_DATA_DIR / "metadata.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file {meta_path} not found. Run scripts/prepare_data.py first.")
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.num_questions = self.metadata['num_questions']
        self.num_tags = self.metadata['num_tags']
        self.num_parts = self.metadata['num_parts']
        self.question_to_idx = self.metadata.get('question_to_idx', {})
        self.tag_to_idx = self.metadata.get('tag_to_idx', {})
        print(f"  Loaded Vocabulary: {self.num_questions:,} Questions | {self.num_tags} Concept Tags | {self.num_parts} Parts")

    def _load_recommendation_catalog(self):
        q_path = PROCESSED_DATA_DIR / "questions_contents.csv"
        l_path = PROCESSED_DATA_DIR / "lectures_contents.csv"
        q_df = pd.read_csv(q_path) if q_path.exists() else pd.DataFrame()
        l_df = pd.read_csv(l_path) if l_path.exists() else pd.DataFrame()
        self.recommender = RecommendationEngine(q_df, l_df)
        print(f"  Recommendation Catalog: {len(q_df):,} Practice Questions | {len(l_df):,} Video Lectures")

    def _load_models(self):
        """Loads pre-trained model weights from checkpoints."""
        print("\n  Loading Pre-Trained Model Checkpoints from models/checkpoints/...")
        t0 = time.time()

        # Model 1: LSTM + Attention
        m1_path = CHECKPOINTS_DIR / "model1_lstm_corrected_best.pth"
        if not m1_path.exists():
            m1_path = CHECKPOINTS_DIR / "model1_lstm_best.pth"
        self.m1 = LSTMAttentionKT(num_questions=self.num_questions, num_parts=self.num_parts, num_tags=self.num_tags).to(self.device)
        self.m1.load_state_dict(torch.load(m1_path, map_location=self.device, weights_only=False))
        self.m1.eval()
        print(f"  [OK] Model 1 (LSTM + Attention) Loaded       : {m1_path.name}")

        # Model 2: Transformer + Knowledge Tracing
        m2_path = CHECKPOINTS_DIR / "model2_transformer_corrected_best.pth"
        if not m2_path.exists():
            m2_path = CHECKPOINTS_DIR / "model2_transformer_best.pth"
        self.m2 = TransformerKT(num_questions=self.num_questions, num_parts=self.num_parts, num_tags=self.num_tags).to(self.device)
        self.m2.load_state_dict(torch.load(m2_path, map_location=self.device, weights_only=False))
        self.m2.eval()
        print(f"  [OK] Model 2 (Transformer + KT) Loaded       : {m2_path.name}")

        # Model 3: BERT-style + NCF
        m3_path = CHECKPOINTS_DIR / "model3_bert_ncf_corrected_best.pth"
        if not m3_path.exists():
            m3_path = CHECKPOINTS_DIR / "model3_bert_ncf_best.pth"
        self.m3 = BERTNCF(num_questions=self.num_questions, num_parts=self.num_parts, num_tags=self.num_tags).to(self.device)
        self.m3.load_state_dict(torch.load(m3_path, map_location=self.device, weights_only=False))
        self.m3.eval()
        print(f"  [OK] Model 3 (BERT-style + NCF) Loaded       : {m3_path.name}")

        # Model 4: Autoencoder + Recommender
        m4_path = CHECKPOINTS_DIR / "model4_autoencoder_corrected_best.pth"
        if not m4_path.exists():
            m4_path = CHECKPOINTS_DIR / "model4_autoencoder_best.pth"
        self.m4 = AutoencoderRecommender(input_dim=500, latent_dim=32, hidden_dim=64, num_resources=500).to(self.device)
        self.m4.load_state_dict(torch.load(m4_path, map_location=self.device, weights_only=False))
        self.m4.eval()
        print(f"  [OK] Model 4 (Autoencoder Recommender) Loaded: {m4_path.name}")

        # Model 5: CNN + LSTM
        m5_path = CHECKPOINTS_DIR / "model5_cnn_lstm_corrected_best.pth"
        if not m5_path.exists():
            m5_path = CHECKPOINTS_DIR / "model5_cnn_lstm_best.pth"
        self.m5 = CNNLSTM(num_questions=self.num_questions, num_parts=self.num_parts, num_tags=self.num_tags).to(self.device)
        self.m5.load_state_dict(torch.load(m5_path, map_location=self.device, weights_only=False))
        self.m5.eval()
        print(f"  [OK] Model 5 (CNN + LSTM) Loaded             : {m5_path.name}")

        print(f"  [READY] All 5 models active in memory ({time.time() - t0:.2f}s). Ready for instant inference!\n")

    def predict_sequence(self, sequence_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes simultaneous multi-model forward pass for a student sequence.
        """
        ds = KTSequenceDataset([sequence_events], max_seq_len=DEFAULT_SEQUENCE_LENGTH)
        inputs, _, mask, _ = ds[0]

        inputs = inputs.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        t_start = time.time()
        with torch.no_grad():
            # Model 1
            logits1 = self.m1(inputs)
            p1 = float(torch.sigmoid(logits1)[0, -1].item())

            # Model 2
            logits2 = self.m2(inputs)
            p2 = float(torch.sigmoid(logits2)[0, -1].item())

            # Model 3
            logits3 = self.m3(inputs)
            p3 = float(torch.sigmoid(logits3)[0, -1].item())

            # Model 5
            logits5 = self.m5(inputs)
            p5 = float(torch.sigmoid(logits5)[0, -1].item())

            # Model-Derived Knowledge State
            model_knowledge_state = self.m1.compute_knowledge_state(inputs, mask)

            # Model 4 (Autoencoder) Profile Scoring
            profile_vec = torch.zeros(1, 500, device=self.device)
            for ev in sequence_events:
                q_idx = ev.get('question_id', 0)
                if q_idx < 500:
                    profile_vec[0, q_idx] = 1.0 if ev.get('is_correct', 0) == 1 else 0.5
            _, rec_scores = self.m4(profile_vec)
            top_rec_indices = torch.topk(rec_scores[0], k=5).indices.cpu().numpy()

        inference_time_ms = (time.time() - t_start) * 1000

        # Consensus Ensemble (Mean probability)
        ensemble_p = float(np.mean([p1, p2, p3, p5]))

        # Format Knowledge State
        formatted_state = {
            cid: {
                'concept_id': cid,
                'concept_name': f"Concept_{cid}",
                'mastery_score': score,
                'predicted_observations': 1,
                'state_type': 'MODEL-DERIVED KNOWLEDGE STATE'
            }
            for cid, score in model_knowledge_state.items()
        }

        # Gap Detection (< 60% mastery)
        gaps = self.kse.detect_learning_gaps(formatted_state, mastery_threshold=0.60, min_attempts=1)

        # Recommendations
        attempted_ids = {str(ev.get('question_id', '')) for ev in sequence_events}
        recommendations = self.recommender.recommend_for_gaps(gaps, top_k=5, exclude_attempted=attempted_ids)

        return {
            'predictions': {
                'Model 1 (LSTM + Attention)': p1,
                'Model 2 (Transformer + KT)': p2,
                'Model 3 (BERT-style + NCF)': p3,
                'Model 5 (CNN + LSTM)': p5,
                'Ensemble Consensus': ensemble_p
            },
            'inference_time_ms': inference_time_ms,
            'knowledge_state': formatted_state,
            'learning_gaps': gaps,
            'recommendations': recommendations,
            'total_events': len(sequence_events)
        }

    def predict_student_by_id(self, student_id: str, split: str = "test") -> Optional[Dict[str, Any]]:
        """
        Loads a student's interaction stream from test.csv or train.csv and runs prediction.
        """
        file_path = PROCESSED_DATA_DIR / f"{split}.csv"
        if not file_path.exists():
            print(f"  [ERROR] {file_path.name} not found.")
            return None

        # Filter student rows
        df = pd.read_csv(file_path)
        student_rows = df[df['user_id'] == student_id]

        if student_rows.empty:
            # Try searching in other splits
            for alt_split in ["val", "train"]:
                alt_path = PROCESSED_DATA_DIR / f"{alt_split}.csv"
                if alt_path.exists():
                    alt_df = pd.read_csv(alt_path)
                    alt_rows = alt_df[alt_df['user_id'] == student_id]
                    if not alt_rows.empty:
                        student_rows = alt_rows
                        split = alt_split
                        break

        if student_rows.empty:
            print(f"  [NOT FOUND] Student '{student_id}' does not exist in dataset.")
            return None

        events = []
        for _, row in student_rows.iterrows():
            tag_raw = str(row.get('tags', row.get('tag_list', '')))
            clean_tags = tag_raw.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
            tags_split = clean_tags.replace(';', ',').split(',')
            tag_list = [int(t.strip()) for t in tags_split if t.strip().isdigit()]
            events.append({
                'question_id': int(row.get('question_idx', 0)),
                'part': int(row.get('part', 0)),
                'tags': tag_list,
                'is_correct': int(row.get('is_correct', 0)),
                'response_time': float(row.get('response_time_norm', 0.0)) if pd.notna(row.get('response_time_norm')) else 0.0,
                'source_encoded': int(row.get('source_encoded', 0)) if pd.notna(row.get('source_encoded')) else 0,
                'platform_encoded': int(row.get('platform_encoded', 0)) if pd.notna(row.get('platform_encoded')) else 0
            })

        result = self.predict_sequence(events)
        result['student_id'] = student_id
        result['split'] = split

        # Also compute historical baseline accuracy state
        baseline_state = self.kse.compute_historical_baseline_state(student_rows)
        result['baseline_state'] = baseline_state

        # If sequence model window didn't have deficits (< 60%), check cumulative historical deficits
        if not result['learning_gaps'] and baseline_state:
            hist_gaps = self.kse.detect_learning_gaps(baseline_state, mastery_threshold=0.60, min_attempts=1)
            if hist_gaps:
                result['learning_gaps'] = hist_gaps
                attempted_ids = {str(ev.get('question_id', '')) for ev in events}
                result['recommendations'] = self.recommender.recommend_for_gaps(hist_gaps, top_k=5, exclude_attempted=attempted_ids)

        return result

    def print_result_dashboard(self, result: Dict[str, Any]):
        """
        Renders an academic dashboard summarizing multi-model results.
        """
        print("\n" + "=" * 80)
        stu_id = result.get('student_id', 'Custom Simulated Student')
        split_info = f" ({result.get('split', '').upper()} SET)" if 'split' in result else ""
        print(f"  PREDICTION DASHBOARD — STUDENT: {stu_id}{split_info}")
        print("=" * 80)
        print(f"  Total Historical Events: {result['total_events']:,} | Simultaneous Inference: {result['inference_time_ms']:.2f} ms")

        # 1. Multi-Model Predictions
        print("\n[1] SIMULTANEOUS MULTI-MODEL PREDICTIONS (P(Correct on Next Question)):")
        print("-" * 80)
        preds = result['predictions']
        for model_name, prob in preds.items():
            bar_len = int(prob * 30)
            bar = "=" * bar_len + "-" * (30 - bar_len)
            is_ensemble = "*" if "Consensus" in model_name else " "
            print(f"  {is_ensemble} {model_name:<30} : {prob*100:5.1f}%  [{bar}]")

        # 2. Diagnosed Knowledge State
        print("\n[2] REAL-TIME KNOWLEDGE STATE & CONCEPT MASTERY (Sample Top 5):")
        print("-" * 80)
        ks = result['knowledge_state']
        if not ks:
            print("  No multi-event concept history detected.")
        else:
            sorted_concepts = sorted(ks.values(), key=lambda x: x['mastery_score'], reverse=True)
            for c in sorted_concepts[:5]:
                m = c['mastery_score']
                print(f"  Concept #{c['concept_id']:<3} ({c['concept_name']:<12}) : Mastery = {m*100:5.1f}%")

        # 3. Learning Gaps
        print("\n[3] DIAGNOSED LEARNING DEFICIENCIES (Concepts with Mastery < 60%):")
        print("-" * 80)
        gaps = result['learning_gaps']
        if not gaps:
            print("  [OK] No critical concept deficiencies detected! Student is performing well above mastery threshold.")
        else:
            print(f"  Identified {len(gaps)} deficit areas. Top critical weaknesses:")
            for i, gap in enumerate(gaps[:3], 1):
                cid = gap.get('concept_id', gap.get('concept', 0))
                m = gap.get('mastery_score', gap.get('mastery', 0.0))
                print(f"    {i}. Concept #{cid:<3} : Mastery = {m*100:5.1f}% (Critical Remedial Priority)")

        # 4. Recommendations
        print("\n[4] TOP PERSONALIZED LEARNING RECOMMENDATIONS (Authentic EdNet Catalog):")
        print("-" * 80)
        recs = result['recommendations']
        if not recs:
            print("  No recommendations generated.")
        else:
            for i, r in enumerate(recs, 1):
                rtype = r.get('type', 'resource').upper()
                rid = r.get('id', 'N/A')
                cid = r.get('concept', 'N/A')
                part = f"Part {r.get('part')}" if r.get('part') else ""
                dur = f"({r.get('duration_sec')}s)" if r.get('duration_sec') else ""
                print(f"    {i}. [{rtype:<11}] ID: {rid:<8} | Targets Concept: {cid:<3} | {part} {dur}")
        print("=" * 80)

    def print_ai_explanation(self, result: Dict[str, Any]):
        """
        Renders Generative AI interpretations of diagnosed learning gaps and recommendations via Ollama.
        """
        try:
            from ollama_ai.learner_context import build_learner_context
            from ollama_ai.response_generator import ResponseGenerator
            ctx = build_learner_context(inference_result=result, student_id=result.get('student_id'))
            gen = ResponseGenerator()
            print("\n[5] GENERATIVE AI PEDAGOGICAL INTERPRETATION (OLLAMA):")
            print("-" * 80)
            status_badge = "OLLAMA ACTIVE" if gen.is_ai_available() else "DETERMINISTIC FALLBACK (Ollama offline)"
            print(f"  AI Status: {status_badge}")
            print("\n  >> AI Explanation of Diagnosed Deficiencies & Recommendations:")
            gap_exp = gen.explain_learning_gaps(ctx)
            print("  " + gap_exp.get('explanation', '').replace("\n", "\n  "))
            print("\n  >> Recommended Pedagogical Sequencing:")
            rec_exp = gen.explain_recommendations(ctx)
            print("  " + rec_exp.get('explanation', '').replace("\n", "\n  "))
            print("=" * 80)
        except Exception as e:
            print(f"  [AI Layer Notice] Generative AI explanation unavailable: {e}")


def interactive_menu(engine: MultiModelInferenceEngine):
    """
    Terminal CLI menu for interactive student evaluation.
    """
    while True:
        print("\n" + "-" * 80)
        print("  INTERACTIVE INFERENCE MENU — SMART EDUCATION")
        print("-" * 80)
        print("  [1] Lookup & Predict Authentic Student by ID (e.g. u1165, u502, u249)")
        print("  [2] Enter Custom Student Performance Sequence (Simulated Input)")
        print("  [3] Run Multi-Student Batch Sanity Benchmark (5 Random Unseen Students)")
        print("  [4] Exit")
        print("  [5] Generative AI Tutor & Diagnostic Interpretation (Ollama)")
        print("-" * 80)

        choice = input(">> Select an option [1-5]: ").strip()


        if choice == '1':
            stu_id = input(">> Enter Student ID (e.g., u1165): ").strip()
            if not stu_id:
                print("  Student ID cannot be empty.")
                continue
            print(f"\n  Fetching events and executing 5-model inference for {stu_id}...")
            res = engine.predict_student_by_id(stu_id)
            if res:
                engine.print_result_dashboard(res)

        elif choice == '2':
            print("\n" + "=" * 60)
            print("  CUSTOM STUDENT INTERACTION INPUT")
            print("=" * 60)
            try:
                num_events = int(input(">> How many practice questions to simulate? (e.g., 3 to 10): ").strip())
            except ValueError:
                print("  Invalid number.")
                continue

            events = []
            for i in range(1, num_events + 1):
                print(f"\n  -- Event #{i} --")
                q_id = input(f"    Question Index (0 to {engine.num_questions-1}, default={i*10}): ").strip()
                q_val = int(q_id) if q_id.isdigit() else i * 10

                tag_id = input(f"    Concept Tag ID (0 to {engine.num_tags-1}, default=18): ").strip()
                tag_val = int(tag_id) if tag_id.isdigit() else 18

                is_corr = input("    Result (1 = Correct, 0 = Incorrect, default=1): ").strip()
                is_corr_val = 1 if is_corr not in ['0', 'f', 'false'] else 0

                resp_time = input("    Response Time in sec (default=15.0): ").strip()
                try:
                    rt_val = float(resp_time)
                except ValueError:
                    rt_val = 15.0

                events.append({
                    'question_id': q_val,
                    'part': 1,
                    'tags': [tag_val],
                    'is_correct': is_corr_val,
                    'response_time': (rt_val - 20.0) / 15.0,  # rough normalization
                    'source_encoded': 1,
                    'platform_encoded': 1
                })

            print("\n  Executing simultaneous 5-model forward pass on custom inputs...")
            res = engine.predict_sequence(events)
            res['student_id'] = "Custom_User_Terminal"
            engine.print_result_dashboard(res)

        elif choice == '3':
            print("\n  Running batch sanity check across 5 random unseen test students...")
            test_csv = PROCESSED_DATA_DIR / "test.csv"
            if not test_csv.exists():
                print("  test.csv not found.")
                continue
            df = pd.read_csv(test_csv)
            sample_uids = df['user_id'].unique()[:5]

            print("\n" + "=" * 85)
            print(f"  {'Student ID':<12} | {'M1 (LSTM)':<10} | {'M2 (Trans)':<10} | {'M3 (BERT)':<10} | {'M5 (CNN)':<10} | {'Consensus':<10} | {'Gaps':<5}")
            print("=" * 85)

            for uid in sample_uids:
                res = engine.predict_student_by_id(uid)
                if res:
                    p = res['predictions']
                    g = len(res['learning_gaps'])
                    print(f"  {uid:<12} | {p['Model 1 (LSTM + Attention)']*100:6.1f}%    | {p['Model 2 (Transformer + KT)']*100:6.1f}%    | {p['Model 3 (BERT-style + NCF)']*100:6.1f}%    | {p['Model 5 (CNN + LSTM)']*100:6.1f}%    | {p['Ensemble Consensus']*100:6.1f}%    | {g:<5}")
            print("=" * 85)

        elif choice == '4':
            print("\n  Exiting Smart Education Inference System. Goodbye!\n")
            break
        elif choice == '5':
            print("\n" + "=" * 60)
            print("  GENERATIVE AI TUTOR & DIAGNOSTIC INTERPRETATION (OLLAMA)")
            print("=" * 60)
            stu_id = input(">> Enter Student ID for Context (e.g. u1165, or press Enter): ").strip()
            res = engine.predict_student_by_id(stu_id) if stu_id else None
            try:
                from ollama_ai.learner_context import build_learner_context
                from ollama_ai.tutor import AITutor
                from ollama_ai.response_generator import ResponseGenerator
                ctx = build_learner_context(inference_result=res, student_id=stu_id or "Custom_Learner")
                gen = ResponseGenerator()
                tutor = AITutor()
                print(f"  AI Service Status: {'ONLINE' if gen.is_ai_available() else 'OFFLINE (Deterministic Fallback Active)'}")
                print("\n  [A] Explain Diagnosed Knowledge Gaps")
                print("  [B] Explain Recommendations")
                print("  [C] Generate Personalized Study Plan")
                print("  [D] Ask AI Tutor a Custom Question")
                sub = input(">> Select an action [A-D]: ").strip().upper()
                if sub == 'A':
                    out = gen.explain_learning_gaps(ctx)
                    print("\n" + out.get('explanation', ''))
                elif sub == 'B':
                    out = gen.explain_recommendations(ctx)
                    print("\n" + out.get('explanation', ''))
                elif sub == 'C':
                    out = gen.generate_study_plan(ctx)
                    print("\n" + out.get('explanation', ''))
                elif sub == 'D':
                    q = input(">> Enter question for AI Tutor: ").strip()
                    if q:
                        out = tutor.ask(ctx, query=q)
                        print("\n" + out.get('response', ''))
                else:
                    print("  Invalid action.")
            except Exception as e:
                print(f"  AI interpretation error: {e}")
        else:
            print("  Invalid selection. Please enter 1, 2, 3, 4, or 5.")


def main():
    parser = argparse.ArgumentParser(description="Smart Education — Master Multi-Model Inference CLI")
    parser.add_argument("--student", type=str, default=None, help="Directly predict for student ID (e.g., u1165)")
    parser.add_argument("--split", type=str, default="test", help="Split to lookup student from ('test', 'train', 'val')")
    parser.add_argument("--demo", action="store_true", help="Run a quick automated demonstration on sample student u1165")
    parser.add_argument("--benchmark", action="store_true", help="Run quick 5-student multi-model comparison")
    parser.add_argument("--ai", action="store_true", help="Include Generative AI explanations via Ollama")
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cuda' or 'cpu')")
    args = parser.parse_args()

    engine = MultiModelInferenceEngine(device=args.device)

    if args.student:
        print(f"\n  Running direct inference for Student: {args.student}...")
        res = engine.predict_student_by_id(args.student, split=args.split)
        if res:
            engine.print_result_dashboard(res)
            if args.ai:
                engine.print_ai_explanation(res)
    elif args.demo:
        print("\n  Running demonstration on sample unseen student u1165...")
        res = engine.predict_student_by_id("u1165", split="test")
        if res:
            engine.print_result_dashboard(res)
            if args.ai:
                engine.print_ai_explanation(res)
    elif args.benchmark:
        print("\n  Running batch sanity benchmark on unseen test students...")
        test_csv = PROCESSED_DATA_DIR / "test.csv"
        df = pd.read_csv(test_csv)
        sample_uids = df['user_id'].unique()[:5]
        print("\n" + "=" * 85)
        print(f"  {'Student ID':<12} | {'M1 (LSTM)':<10} | {'M2 (Trans)':<10} | {'M3 (BERT)':<10} | {'M5 (CNN)':<10} | {'Consensus':<10} | {'Gaps':<5}")
        print("=" * 85)
        for uid in sample_uids:
            res = engine.predict_student_by_id(uid)
            if res:
                p = res['predictions']
                g = len(res['learning_gaps'])
                print(f"  {uid:<12} | {p['Model 1 (LSTM + Attention)']*100:6.1f}%    | {p['Model 2 (Transformer + KT)']*100:6.1f}%    | {p['Model 3 (BERT-style + NCF)']*100:6.1f}%    | {p['Model 5 (CNN + LSTM)']*100:6.1f}%    | {p['Ensemble Consensus']*100:6.1f}%    | {g:<5}")
        print("=" * 85)
    else:
        # Launch Interactive Menu
        interactive_menu(engine)


if __name__ == "__main__":
    main()

