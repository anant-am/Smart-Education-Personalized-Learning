"""
Smart Education Project — Unified Data Preparation & Leakage Verification
==========================================================================
Academic Rigor Standards:
1. Supports Execution Modes:
   - SMOKE TEST MODE (--mode smoke): Tiny subset (25 users) to verify complete pipeline.
   - DEVELOPMENT MODE (--mode dev): 1,000 users for development and rapid validation.
   - FULL MODE (--mode full): Complete EdNet-KT3 (297,915 users) with memory-safe processing.
2. Performs strict USER-LEVEL 80:20 SPLIT:
   - 80% USERS -> Training / Development
   - 20% USERS -> Frozen Final Test
   - Zero student overlap verified: set(train_users) ∩ set(test_users) == empty.
3. Derives ground-truth correctness strictly by joining student user_answer with correct_answer from questions.csv.
4. Fits continuous scalers and item vocabularies EXCLUSIVELY on training users.
5. Transforms frozen test cohort with frozen statistics (novel items mapped to 0).
6. Executes all 7 required leakage checks (A through G) before permitting downstream modeling.
"""

import sys
import os
import time
import pickle
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EDNET_CONTENTS_DIR, EDNET_KT3_FULL_DIR, EDNET_KT3_DEV_DIR, EDNET_KT3_TEST_DIR,
    PROCESSED_DATA_DIR, REPORTS_DIR, RANDOM_SEED, EXECUTION_MODE, ensure_dirs
)
from src.data_pipeline import EdNetDataPipeline
from src.leakage_checks import LeakageAuditor

ensure_dirs()
np.random.seed(RANDOM_SEED)


def main(mode: Optional[str] = None, max_users: Optional[int] = None, force_reprocess: bool = False):
    parser = argparse.ArgumentParser(description="Smart Education Data Preparation & Leakage Audit")
    parser.add_argument("--mode", type=str, default=mode or EXECUTION_MODE or "dev",
                        choices=["smoke", "dev", "full"], help="Pipeline execution mode")
    parser.add_argument("--max-users", type=int, default=max_users,
                        help="Override number of total users to process")
    parser.add_argument("--force", action="store_true", default=force_reprocess,
                        help="Force reprocessing even if existing files exist")
    args, _ = parser.parse_known_args()

    active_mode = args.mode.lower().strip()
    total_start_time = time.time()

    print("=" * 85)
    print("  SMART EDUCATION — DATA PREPARATION & LEAKAGE VERIFICATION PIPELINE")
    print(f"  Execution Mode: {active_mode.upper()} | Random Seed: {RANDOM_SEED}")
    print("=" * 85)

    pipeline = EdNetDataPipeline()

    # Step 1: Ingest Contents Metadata
    print("\n[Step 1] Ingesting EdNet Contents Metadata (questions.csv & lectures.csv)...")
    questions_df, lectures_df, q_lookup, l_lookup = pipeline.load_contents_metadata()
    print(f"  Loaded {len(questions_df):,} questions and {len(lectures_df):,} lectures.")

    # Determine user target directory
    if active_mode == "full" and EDNET_KT3_FULL_DIR.exists():
        target_dir = EDNET_KT3_FULL_DIR
    elif EDNET_KT3_DEV_DIR.exists():
        target_dir = EDNET_KT3_DEV_DIR
    elif EDNET_KT3_FULL_DIR.exists():
        target_dir = EDNET_KT3_FULL_DIR
    else:
        target_dir = EDNET_KT3_DEV_DIR

    print(f"  Source KT3 Directory: {target_dir}")
    all_user_files = pipeline.discover_users(target_dir)
    actual_user_count = len(all_user_files)
    print(f"  Discovered actual student file count: {actual_user_count:,}")

    # Determine cohort size for active mode
    if args.max_users is not None and args.max_users > 0:
        cohort_files = all_user_files[:args.max_users]
    elif active_mode == "smoke":
        cohort_files = all_user_files[:25]
    elif active_mode == "dev":
        cohort_files = all_user_files[:min(1000, len(all_user_files))]
    else:  # full
        cohort_files = all_user_files

    print(f"  Selected cohort size for '{active_mode}' mode: {len(cohort_files):,} students.")

    # Step 2: Strict USER-LEVEL 80:20 Split + Intra-Dev Split
    print("\n[Step 2] Executing Strict USER-LEVEL 80:20 Split...")
    train_dev_files, test_files = pipeline.split_users_80_20(cohort_files, train_ratio=0.8, seed=RANDOM_SEED)

    train_dev_uids = set([p.stem for p in train_dev_files])
    test_uids = set([p.stem for p in test_files])

    # Enforce zero overlap assertion between train/dev and test
    overlap = train_dev_uids.intersection(test_uids)
    if len(overlap) > 0:
        raise RuntimeError(f"FATAL: User overlap detected! {len(overlap)} students appear in both partitions.")
    print(f"  [DISJOINTNESS CHECK PASSED] Train/Dev: {len(train_dev_uids):,} users | Frozen Test: {len(test_uids):,} users | Overlap: 0")

    # Intra-development split for validation strategy 1 (80% train, 20% val of dev users)
    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_dev = rng.permutation(len(train_dev_files))
    n_dev_train = max(1, int(len(train_dev_files) * 0.8))
    train_files = [train_dev_files[i] for i in shuffled_dev[:n_dev_train]]
    val_files = [train_dev_files[i] for i in shuffled_dev[n_dev_train:]]
    dev_train_uids = set([p.stem for p in train_files])
    dev_val_uids = set([p.stem for p in val_files])

    intra_overlap = dev_train_uids.intersection(dev_val_uids)
    if len(intra_overlap) > 0:
        raise RuntimeError(f"FATAL: Intra-development overlap! {len(intra_overlap)} users in train and val.")
    print(f"  [INTRA-DEV SPLIT PASSED] Train Users: {len(dev_train_uids):,} | Val Users: {len(dev_val_uids):,} | Overlap: 0")

    # Step 3: Process interactions for Train, Val, and Test
    print("\n[Step 3] Reconstructing and preprocessing learning events from raw KT3 files...")
    train_q, dev_l, dev_e, _ = pipeline.process_cohort_files(
        train_files, f"Training Cohort ({len(train_files):,} users)", q_lookup, l_lookup
    )
    val_q, _, _, _ = pipeline.process_cohort_files(
        val_files, f"Validation Cohort ({len(val_files):,} users)", q_lookup, l_lookup
    )
    test_q, test_l, test_e, _ = pipeline.process_cohort_files(
        test_files, f"Frozen Test Cohort ({len(test_files):,} users)", q_lookup, l_lookup
    )

    print(f"  Extracted question events: Train = {len(train_q):,} | Val = {len(val_q):,} | Frozen Test = {len(test_q):,}")

    # Step 4: Fit scalers & vocabularies on Train ONLY, transform val and test with frozen parameters
    print("\n[Step 4] Fitting Vocabularies & Scalers Exclusively on Training Partition...")
    train_df, metadata = pipeline.fit_and_transform_dev(train_q)
    val_df = pipeline.transform_test_cohort(val_q, metadata)
    test_df = pipeline.transform_test_cohort(test_q, metadata)
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    # Step 5: Verify zero user overlap across all partitions
    print("\n[Step 5] Verifying User-Level Disjointness Across All Partitions...")
    assert len(set(train_df['user_id']).intersection(set(test_df['user_id']))) == 0, "Train-Test user overlap!"
    assert len(set(val_df['user_id']).intersection(set(test_df['user_id']))) == 0, "Val-Test user overlap!"
    print("  [ALL PARTITIONS DISJOINT] Train intersect Val = empty, Dev intersect Test = empty. Zero leakage verified.")

    # Step 6: Save clean processed datasets
    print("\n[Step 6] Persisting Clean Processed Datasets & Metadata...")
    train_dev_csv = PROCESSED_DATA_DIR / "train_dev.csv"
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    val_csv = PROCESSED_DATA_DIR / "val.csv"
    test_csv = PROCESSED_DATA_DIR / "test.csv"
    meta_path = PROCESSED_DATA_DIR / "metadata.pkl"

    dev_df.to_csv(train_dev_csv, index=False)
    dev_df.to_csv(PROCESSED_DATA_DIR / "question_events.csv", index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    questions_df.to_csv(PROCESSED_DATA_DIR / "questions_contents.csv", index=False)
    lectures_df.to_csv(PROCESSED_DATA_DIR / "lectures_contents.csv", index=False)

    if dev_l:
        pd.DataFrame(dev_l).to_csv(PROCESSED_DATA_DIR / "lecture_events.csv", index=False)
    if dev_e:
        pd.DataFrame(dev_e).to_csv(PROCESSED_DATA_DIR / "explanation_events.csv", index=False)

    print(f"  Saved: train_dev.csv ({len(dev_df):,} rows, {dev_df['user_id'].nunique():,} users)")
    print(f"  Saved: train.csv     ({len(train_df):,} rows, {train_df['user_id'].nunique():,} users)")
    print(f"  Saved: val.csv       ({len(val_df):,} rows, {val_df['user_id'].nunique():,} users)")
    print(f"  Saved: test.csv      ({len(test_df):,} rows, {test_df['user_id'].nunique():,} users)")
    print(f"  Saved: metadata.pkl  ({metadata['num_questions']:,} questions, {metadata['num_tags']} tags)")

    # Step 7: Comprehensive Leakage Auditing (Checks A through G)
    print("\n[Step 7] Executing Comprehensive Data Leakage Audits (Checks A through G)...")
    auditor = LeakageAuditor()

    # Check A: User Leakage
    auditor.check_user_leakage(train_dev_uids, test_uids)

    # Check B: Duplicate Leakage
    auditor.check_duplicate_leakage(dev_df)

    # Check C: Target Leakage
    feat_cols = [
        'previous_accuracy', 'recent_accuracy_5', 'attempt_count',
        'response_time_norm', 'time_since_prev_norm',
        'source_encoded', 'platform_encoded', 'part', 'question_idx'
    ]
    auditor.check_target_leakage(dev_df, feat_cols)

    # Check D: Temporal Leakage
    auditor.check_temporal_leakage(dev_df)

    # Check E: Metadata Leakage
    auditor.check_metadata_leakage(feat_cols)

    # Check F: Target-Derived Labels
    auditor.check_target_derived_labels(feat_cols, target_col='is_correct')

    # Check G: Preprocessing Leakage
    auditor.check_preprocessing_leakage(train_df, test_df, metadata)

    # Additional: Resource Catalog Leakage
    top_candidates = list(metadata['question_to_idx'].keys())[:100]
    auditor.check_resource_leakage(dev_df, test_df, top_candidates)

    report_content = auditor.generate_report()
    all_leakage_passed = all(status for status, _ in auditor.results.values())

    for cat, info in auditor.check_details.items():
        status_tag = "PASS" if info['passed'] else "FAIL"
        print(f"  [{status_tag}] {cat}: Violations = {info['count']} | {info['evidence']}")

    if not all_leakage_passed:
        raise RuntimeError("FATAL: Data leakage detected! Pipeline halted.")

    print(f"\n  Saved Leakage Report to: {REPORTS_DIR / 'leakage_report.md'}")

    # Step 8: Generate Audit & Split Reports
    print("\n[Step 8] Generating Empirical Data Audit & Split Reports...")
    pipeline.generate_data_audit_and_split_reports(
        dev_df, test_df, train_df, val_df, list(train_dev_uids), list(test_uids), metadata
    )

    elapsed_sec = time.time() - total_start_time
    print(f"\n  DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_sec:.2f}s!")
    print("=" * 85)

    return {
        'actual_user_count': actual_user_count,
        'cohort_users': len(cohort_files),
        'train_dev_users': len(train_dev_uids),
        'test_users': len(test_uids),
        'train_events': len(train_df),
        'val_events': len(val_df),
        'test_events': len(test_df),
        'leakage_passed': all_leakage_passed,
        'elapsed_sec': elapsed_sec
    }


if __name__ == "__main__":
    main()
