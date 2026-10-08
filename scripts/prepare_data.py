"""
Smart Education Project — Data Preparation Orchestration Script
================================================================
Entrypoint script for Phase 2 & 3:
1. Reconstructs EdNet learning events from raw KT3 interactions for:
   - Development cohort (KT-3 1000)
   - Final Unseen Test cohort (KT-3 250 TEST)
2. Fits vocabularies and continuous feature normalizers exclusively on Dev.
3. Transforms Unseen Test cohort using frozen parameters.
4. Generates chronological train/val split of Development cohort.
5. Runs LeakageAuditor and writes:
   - reports/data_audit_report.md
   - reports/leakage_report.md
   - reports/split_report.md
"""

import sys
import os
import time
import pickle
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EDNET_CONTENTS_DIR, EDNET_KT3_DEV_DIR, EDNET_KT3_TEST_DIR,
    PROCESSED_DATA_DIR, REPORTS_DIR, RANDOM_SEED, ensure_dirs
)
from src.data_pipeline import EdNetDataPipeline
from src.leakage_checks import LeakageAuditor

ensure_dirs()


def main():
    print("=" * 80)
    print("  PHASE 2/3: SHARED DATA PIPELINE & COMPREHENSIVE LEAKAGE AUDIT")
    print("=" * 80)

    start_time = time.time()
    pipeline = EdNetDataPipeline()

    # 1. Load contents
    print("\n[Step 1] Ingesting EdNet Contents Metadata...")
    questions_df, lectures_df, q_lookup, l_lookup = pipeline.load_contents_metadata()
    print(f"  Loaded {len(questions_df):,} questions and {len(lectures_df):,} lectures.")

    # 2. Check if raw reconstructed events already exist to save processing time
    dev_csv_path = PROCESSED_DATA_DIR / "train_dev.csv"
    test_csv_path = PROCESSED_DATA_DIR / "test_unseen_250.csv"
    meta_path = PROCESSED_DATA_DIR / "metadata.pkl"

    if dev_csv_path.exists() and test_csv_path.exists() and meta_path.exists():
        print("\n[Step 2] Found existing preprocessed Dev and Test datasets in data/processed/.")
        print("  Loading preprocessed tables for verification and auditing...")
        dev_df = pd.read_csv(dev_csv_path)
        test_df = pd.read_csv(test_csv_path)
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        dev_users = sorted(dev_df['user_id'].unique())
        test_users = sorted(test_df['user_id'].unique())
    else:
        print("\n[Step 2] Reconstructing educational interactions from raw KT-3 files...")
        dev_q, dev_l, dev_e, dev_users = pipeline.process_cohort_directory(
            EDNET_KT3_DEV_DIR, "Development Cohort (1000 users)", q_lookup, l_lookup
        )
        test_q, test_l, test_e, test_users = pipeline.process_cohort_directory(
            EDNET_KT3_TEST_DIR, "Final Unseen Test Cohort (250 users)", q_lookup, l_lookup
        )

        dev_df, metadata = pipeline.fit_and_transform_dev(dev_q)
        test_df = pipeline.transform_test_cohort(test_q, metadata)

        dev_df.to_csv(dev_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

    print(f"  Development Cohort: {len(dev_df):,} interactions ({dev_df['user_id'].nunique()} students)")
    print(f"  Final Test Cohort:  {len(test_df):,} interactions ({test_df['user_id'].nunique()} students)")

    # 3. Chronological 80/20 train/val split on Development
    print("\n[Step 3] Chronological Train/Val Splitting of Development Cohort...")
    train_df, val_df = pipeline.split_development_chronologically(dev_df, train_ratio=0.8)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    # The 250 unseen test cohort is preserved as test.csv
    test_df.to_csv(test_path, index=False)

    print(f"  Saved train.csv ({len(train_df):,} rows)")
    print(f"  Saved val.csv   ({len(val_df):,} rows)")
    print(f"  Saved test.csv  ({len(test_df):,} rows)")

    # Save contents copies
    questions_df.to_csv(PROCESSED_DATA_DIR / "questions_contents.csv", index=False)
    lectures_df.to_csv(PROCESSED_DATA_DIR / "lectures_contents.csv", index=False)

    # 4. Comprehensive Leakage Auditing
    print("\n[Step 4] Executing Comprehensive Leakage Audits...")
    auditor = LeakageAuditor()

    # Check target leakage
    feat_cols = [
        'previous_accuracy', 'recent_accuracy_5', 'attempt_count',
        'response_time_norm', 'time_since_prev_norm',
        'source_encoded', 'platform_encoded', 'part', 'question_idx'
    ]
    t_pass, t_ev = auditor.check_target_leakage(dev_df, feat_cols)
    print(f"  Target Leakage:        {'PASS' if t_pass else 'FAIL'} | {t_ev}")

    # Check user leakage
    u_pass, u_ev = auditor.check_user_leakage(set(dev_users), set(test_users))
    print(f"  User Cohort Leakage:   {'PASS' if u_pass else 'FAIL'} | {u_ev}")

    # Check duplicate leakage
    d_pass, d_ev = auditor.check_duplicate_leakage(dev_df)
    print(f"  Duplicate Event Check: {'PASS' if d_pass else 'FAIL'} | {d_ev}")

    # Check temporal leakage
    temp_pass, temp_ev = auditor.check_temporal_leakage(dev_df)
    print(f"  Temporal Monotonicity: {'PASS' if temp_pass else 'FAIL'} | {temp_ev}")

    # Check preprocessing leakage
    p_pass, p_ev = auditor.check_preprocessing_leakage(dev_df, test_df, metadata)
    print(f"  Preprocessing Bounds:  {'PASS' if p_pass else 'FAIL'} | {p_ev}")

    # Check resource leakage
    top_candidates = list(metadata['question_to_idx'].keys())[:100]
    r_pass, r_ev = auditor.check_resource_leakage(dev_df, test_df, top_candidates)
    print(f"  Resource Isolation:    {'PASS' if r_pass else 'FAIL'} | {r_ev}")

    auditor.generate_report()
    print("  Saved: reports/leakage_report.md")

    # 5. Generate Audit & Split Reports
    print("\n[Step 5] Writing Academic Documentation Reports...")
    pipeline.generate_data_audit_and_split_reports(
        dev_df, test_df, train_df, val_df, dev_users, test_users, metadata
    )

    print(f"\n  DATA PREPARATION PIPELINE COMPLETED IN {time.time() - start_time:.2f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
