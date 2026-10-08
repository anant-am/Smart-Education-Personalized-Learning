"""
Step 2 — Rigorous Academic Data Preprocessing
===============================================
Reconstructs meaningful learning events from raw KT3 interactions for both:
1. Development Cohort: A:\\edge download\\EdNet-KT3\\KT-3 (1000) (1,000 users)
2. Final Unseen Test Cohort: A:\\edge download\\EdNet-KT3\\KT-3 (250) TEST (250 users)

CRITICAL ACADEMIC COMPLIANCE:
- All vocabularies (questions, tags, platforms, sources) and continuous normalizers
  (response_time_ms mean/std, time_since_prev_ms mean/std) are fitted EXCLUSIVELY on
  the Development Cohort.
- The 250 Unseen Test Cohort is strictly transformed using the frozen development parameters,
  with unobserved items/tags mapped safely to index 0 (padding/unknown).
- Complete isolation: Zero data leakage between Development and Final Test.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import glob
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)

# ---- Project config ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EDNET_CONTENTS_DIR, EDNET_KT3_DEV_DIR, EDNET_KT3_TEST_DIR,
    PROCESSED_DATA_DIR, RANDOM_SEED, ensure_dirs
)

np.random.seed(RANDOM_SEED)
ensure_dirs()


# =========================================================
# 1. Load Contents Metadata
# =========================================================
def load_contents():
    """Load questions.csv and lectures.csv from EdNet Contents."""
    print("=" * 70)
    print("  STEP 2-A: Loading Contents Metadata")
    print("=" * 70)

    questions_path = EDNET_CONTENTS_DIR / "questions.csv"
    lectures_path = EDNET_CONTENTS_DIR / "lectures.csv"

    questions_df = pd.read_csv(questions_path)
    lectures_df = pd.read_csv(lectures_path)

    print(f"  questions.csv: {len(questions_df):,} rows")
    print(f"  lectures.csv:  {len(lectures_df):,} rows")

    # Build lookup dicts for fast access
    question_lookup = {}
    for _, row in questions_df.iterrows():
        question_lookup[row['question_id']] = {
            'bundle_id': row['bundle_id'],
            'explanation_id': row['explanation_id'],
            'correct_answer': str(row['correct_answer']).strip().lower(),
            'part': int(row['part']),
            'tags': str(row['tags']),
        }

    lecture_lookup = {}
    for _, row in lectures_df.iterrows():
        lecture_lookup[row['lecture_id']] = {
            'part': int(row['part']),
            'tags': str(row['tags']),
            'video_length': int(row['video_length']),
        }

    print(f"  Question lookup: {len(question_lookup):,} entries")
    print(f"  Lecture lookup:  {len(lecture_lookup):,} entries")

    return questions_df, lectures_df, question_lookup, lecture_lookup


# =========================================================
# 2. Reconstruct Learning Events from Raw KT3
# =========================================================
def reconstruct_events_for_user(df_user, user_id, question_lookup, lecture_lookup):
    """
    Reconstruct meaningful learning events from raw KT3 events.
    Question attempt: enter(bundle) -> respond(question) [possibly multiple] -> submit(bundle)
    Lecture session:  enter(lecture) -> quit(lecture)
    Explanation session: enter(explanation) -> quit(explanation)
    """
    question_events = []
    lecture_events = []
    explanation_events = []

    # Sort by timestamp chronologically
    df_user = df_user.sort_values('timestamp').reset_index(drop=True)

    # State tracking
    current_bundle = None
    bundle_enter_ts = None
    responses_in_bundle = []  # list of (question_id, answer, timestamp)
    current_lecture = None
    lecture_enter_ts = None
    current_explanation = None
    explanation_enter_ts = None

    for _, row in df_user.iterrows():
        ts = int(row['timestamp'])
        action = row['action_type']
        item_id = str(row['item_id'])
        source = str(row['source']) if pd.notna(row['source']) else 'unknown'
        user_answer = str(row['user_answer']).strip().lower() if pd.notna(row['user_answer']) and str(row['user_answer']).strip() != '' else None
        platform = str(row['platform']) if pd.notna(row['platform']) else 'unknown'

        prefix = ''.join(c for c in item_id if c.isalpha())

        if action == 'enter':
            if prefix == 'b':
                # New bundle (question group) - save any pending
                if current_bundle and responses_in_bundle:
                    _finalize_question_event(
                        question_events, user_id, current_bundle,
                        bundle_enter_ts, ts, responses_in_bundle,
                        source, platform, question_lookup
                    )
                current_bundle = item_id
                bundle_enter_ts = ts
                responses_in_bundle = []
            elif prefix == 'l':
                current_lecture = item_id
                lecture_enter_ts = ts
            elif prefix == 'e':
                current_explanation = item_id
                explanation_enter_ts = ts

        elif action == 'respond':
            if prefix == 'q' and user_answer:
                responses_in_bundle.append((item_id, user_answer, ts))

        elif action == 'submit':
            if prefix == 'b' and current_bundle == item_id and responses_in_bundle:
                _finalize_question_event(
                    question_events, user_id, current_bundle,
                    bundle_enter_ts, ts, responses_in_bundle,
                    source, platform, question_lookup
                )
                current_bundle = None
                bundle_enter_ts = None
                responses_in_bundle = []

        elif action == 'quit':
            if prefix == 'l' and current_lecture == item_id and lecture_enter_ts:
                duration_ms = ts - lecture_enter_ts
                if duration_ms > 0:
                    linfo = lecture_lookup.get(item_id, {})
                    lecture_events.append({
                        'user_id': user_id,
                        'lecture_id': item_id,
                        'enter_ts': lecture_enter_ts,
                        'quit_ts': ts,
                        'duration_ms': duration_ms,
                        'source': source,
                        'platform': platform,
                        'part': linfo.get('part', -1),
                        'tags': str(linfo.get('tags', '')),
                        'video_length': linfo.get('video_length', -1),
                    })
                current_lecture = None
                lecture_enter_ts = None
            elif prefix == 'e' and current_explanation == item_id and explanation_enter_ts:
                duration_ms = ts - explanation_enter_ts
                if duration_ms > 0:
                    explanation_events.append({
                        'user_id': user_id,
                        'explanation_id': item_id,
                        'enter_ts': explanation_enter_ts,
                        'quit_ts': ts,
                        'duration_ms': duration_ms,
                        'source': source,
                        'platform': platform,
                    })
                current_explanation = None
                explanation_enter_ts = None

    # Finalize any remaining open bundle
    if current_bundle and responses_in_bundle:
        _finalize_question_event(
            question_events, user_id, current_bundle,
            bundle_enter_ts, None, responses_in_bundle,
            source, platform, question_lookup
        )

    return question_events, lecture_events, explanation_events


def _finalize_question_event(events_list, user_id, bundle_id, enter_ts, submit_ts,
                             responses, source, platform, question_lookup):
    """Create a finalized question event from the last response in a bundle."""
    if not responses:
        return

    # Last response is the final submitted answer
    final_q_id, final_answer, final_respond_ts = responses[-1]

    q_info = question_lookup.get(final_q_id, None)
    if q_info is None:
        return  # Question not found in Contents

    correct_answer = q_info['correct_answer']
    is_correct = 1 if final_answer == correct_answer else 0

    # Response time: time from enter to final respond
    response_time_ms = final_respond_ts - enter_ts if enter_ts else 0
    if response_time_ms < 0:
        response_time_ms = 0

    events_list.append({
        'user_id': user_id,
        'question_id': final_q_id,
        'bundle_id': bundle_id,
        'user_answer': final_answer,
        'correct_answer': correct_answer,
        'is_correct': is_correct,
        'part': q_info['part'],
        'tags': q_info['tags'],
        'enter_ts': enter_ts,
        'respond_ts': final_respond_ts,
        'submit_ts': submit_ts,
        'response_time_ms': response_time_ms,
        'num_responses': len(responses),
        'source': source,
        'platform': platform,
    })


def process_user_directory(folder_path, cohort_name, question_lookup, lecture_lookup):
    """Reconstruct events for all users in a given folder."""
    csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
    print(f"\n  Processing {cohort_name} ({len(csv_files)} files in {folder_path.name})...")

    all_q, all_l, all_e = [], [], []
    users_processed = 0
    users_skipped = 0

    start_time = time.time()
    for fpath in tqdm(csv_files, desc=f"  {cohort_name}"):
        user_id = os.path.basename(fpath).replace('.csv', '')
        try:
            df_user = pd.read_csv(fpath)
        except Exception:
            users_skipped += 1
            continue

        q_events, l_events, e_events = reconstruct_events_for_user(
            df_user, user_id, question_lookup, lecture_lookup
        )
        all_q.extend(q_events)
        all_l.extend(l_events)
        all_e.extend(e_events)
        users_processed += 1

    elapsed = time.time() - start_time
    print(f"  [{cohort_name}] Completed in {elapsed:.1f}s | Users: {users_processed} (skipped: {users_skipped})")
    print(f"  [{cohort_name}] Question Events: {len(all_q):,} | Lectures: {len(all_l):,} | Explanations: {len(all_e):,}")

    return all_q, all_l, all_e


# =========================================================
# 3. Feature Derivation & Scaling
# =========================================================
def parse_tags(tag_str):
    try:
        return [int(t) for t in str(tag_str).split(';') if t.strip()]
    except (ValueError, AttributeError):
        return []


def derive_sequential_features(df):
    """Derive lag features per user chronologically without lookahead."""
    df = df.sort_values(['user_id', 'enter_ts']).reset_index(drop=True)
    N = len(df)
    prev_acc_all = np.zeros(N, dtype=np.float32)
    recent5_all = np.zeros(N, dtype=np.float32)
    attempt_count_all = np.zeros(N, dtype=np.int32)
    time_diffs_all = np.zeros(N, dtype=np.float32)

    user_ids = df['user_id'].values
    corrects = df['is_correct'].values
    timestamps = df['enter_ts'].values

    change_indices = np.where(user_ids[:-1] != user_ids[1:])[0] + 1
    split_indices = np.concatenate(([0], change_indices, [N]))

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        user_corrects = corrects[start:end]
        user_ts = timestamps[start:end]
        n_user = end - start

        # Cumulative previous accuracy
        cum_c = np.cumsum(user_corrects)
        cum_cnt = np.arange(1, n_user + 1)
        prev_acc = np.zeros(n_user, dtype=np.float32)
        if n_user > 1:
            prev_acc[1:] = cum_c[:-1] / cum_cnt[:-1]
        prev_acc_all[start:end] = prev_acc

        # Recent 5 moving accuracy
        rec5 = np.zeros(n_user, dtype=np.float32)
        for j in range(1, n_user):
            w_start = max(0, j - 5)
            rec5[j] = user_corrects[w_start:j].mean()
        recent5_all[start:end] = rec5

        # Attempt count
        attempt_count_all[start:end] = cum_cnt - 1

        # Inter-event time
        td = np.zeros(n_user, dtype=np.float32)
        if n_user > 1:
            td[1:] = user_ts[1:] - user_ts[:-1]
        time_diffs_all[start:end] = td

    df['previous_accuracy'] = prev_acc_all
    df['recent_accuracy_5'] = recent5_all
    df['attempt_count'] = attempt_count_all
    df['time_since_prev_ms'] = time_diffs_all
    return df


def fit_and_transform_dev(q_events):
    """Fit encoders/scalers strictly on Development events, then transform."""
    print("\n" + "=" * 70)
    print("  STEP 2-B: Fitting Encoders & Scalers Exclusively on Development Cohort")
    print("=" * 70)

    q_df = pd.DataFrame(q_events)
    q_df = derive_sequential_features(q_df)

    # Categorical encoders
    source_cats = sorted(q_df['source'].unique())
    platform_cats = sorted(q_df['platform'].unique())
    source_map = {s: i + 1 for i, s in enumerate(source_cats)}  # 0 = unknown
    platform_map = {p: i + 1 for i, p in enumerate(platform_cats)}

    q_df['source_encoded'] = q_df['source'].map(lambda s: source_map.get(s, 0))
    q_df['platform_encoded'] = q_df['platform'].map(lambda p: platform_map.get(p, 0))

    # Tags
    q_df['tag_list'] = q_df['tags'].apply(parse_tags)
    all_tags = set()
    for tl in q_df['tag_list']:
        all_tags.update(tl)
    tag_to_idx = {t: i + 1 for i, t in enumerate(sorted(all_tags))}  # 0 = padding
    num_tags = len(tag_to_idx) + 1

    # Questions
    unique_questions = sorted(q_df['question_id'].unique())
    question_to_idx = {q: i + 1 for i, q in enumerate(unique_questions)}  # 0 = padding
    num_questions = len(question_to_idx) + 1
    q_df['question_idx'] = q_df['question_id'].map(lambda q: question_to_idx.get(q, 0))

    # Scalers
    rt = q_df['response_time_ms'].copy()
    rt_clip = float(rt.quantile(0.99))
    rt_clipped = rt.clip(lower=0, upper=rt_clip)
    rt_mean = float(rt_clipped.mean())
    rt_std = float(rt_clipped.std()) if rt_clipped.std() > 0 else 1.0
    q_df['response_time_norm'] = (rt_clipped - rt_mean) / rt_std

    tsp = q_df['time_since_prev_ms'].copy()
    tsp_clip = float(tsp.quantile(0.99))
    tsp_clipped = tsp.clip(lower=0, upper=tsp_clip)
    tsp_mean = float(tsp_clipped.mean())
    tsp_std = float(tsp_clipped.std()) if tsp_clipped.std() > 0 else 1.0
    q_df['time_since_prev_norm'] = (tsp_clipped - tsp_mean) / tsp_std

    metadata = {
        'source_map': source_map,
        'platform_map': platform_map,
        'tag_to_idx': tag_to_idx,
        'question_to_idx': question_to_idx,
        'num_tags': num_tags,
        'num_questions': num_questions,
        'num_parts': 8,
        'rt_clip': rt_clip,
        'rt_mean': rt_mean,
        'rt_std': rt_std,
        'tsp_clip': tsp_clip,
        'tsp_mean': tsp_mean,
        'tsp_std': tsp_std,
    }

    print(f"  Fitted Vocabulary: {num_questions:,} questions, {num_tags} tags")
    print(f"  Response Time Scaler: mean={rt_mean:.1f}ms, std={rt_std:.1f}ms (clip={rt_clip:.1f}ms)")
    print(f"  Inter-Event Scaler:   mean={tsp_mean:.1f}ms, std={tsp_std:.1f}ms (clip={tsp_clip:.1f}ms)")
    return q_df, metadata


def transform_test_cohort(test_q_events, meta):
    """Transform the 250 Unseen Test cohort using strictly frozen Development parameters."""
    print("\n" + "=" * 70)
    print("  STEP 2-C: Transforming 250-User Unseen Test Cohort with Frozen Stats")
    print("=" * 70)

    t_df = pd.DataFrame(test_q_events)
    t_df = derive_sequential_features(t_df)

    # Transform categoricals with fallback to 0
    t_df['source_encoded'] = t_df['source'].map(lambda s: meta['source_map'].get(s, 0))
    t_df['platform_encoded'] = t_df['platform'].map(lambda p: meta['platform_map'].get(p, 0))

    # Transform tags and questions
    t_df['tag_list'] = t_df['tags'].apply(parse_tags)
    t_df['question_idx'] = t_df['question_id'].map(lambda q: meta['question_to_idx'].get(q, 0))

    # Apply frozen scalers
    rt = t_df['response_time_ms'].clip(lower=0, upper=meta['rt_clip'])
    t_df['response_time_norm'] = (rt - meta['rt_mean']) / meta['rt_std']

    tsp = t_df['time_since_prev_ms'].clip(lower=0, upper=meta['tsp_clip'])
    t_df['time_since_prev_norm'] = (tsp - meta['tsp_mean']) / meta['tsp_std']

    unseen_q = (t_df['question_idx'] == 0).sum()
    print(f"  Test Cohort Transformed: {len(t_df):,} events across {t_df['user_id'].nunique()} students")
    print(f"  Unseen Questions (mapped to idx 0): {unseen_q:,} ({unseen_q/len(t_df)*100:.2f}%)")
    return t_df


# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    total_start = time.time()
    questions_df, lectures_df, question_lookup, lecture_lookup = load_contents()

    # 1. Process Development Cohort (1000 users)
    dev_q, dev_l, dev_e = process_user_directory(
        EDNET_KT3_DEV_DIR, "Development Cohort (1000 users)", question_lookup, lecture_lookup
    )

    # 2. Process Final Unseen Test Cohort (250 users)
    test_q, test_l, test_e = process_user_directory(
        EDNET_KT3_TEST_DIR, "Final Unseen Test Cohort (250 users)", question_lookup, lecture_lookup
    )

    # 3. Fit on Dev and Transform both
    dev_df, metadata = fit_and_transform_dev(dev_q)
    test_df = transform_test_cohort(test_q, metadata)

    # 4. Save clean processed datasets
    print("\n" + "=" * 70)
    print("  STEP 2-D: Saving Clean Datasets & Metadata")
    print("=" * 70)

    # Development cohort
    dev_df.to_csv(PROCESSED_DATA_DIR / "train_dev.csv", index=False)
    dev_df.to_csv(PROCESSED_DATA_DIR / "question_events.csv", index=False)
    print(f"  Saved: train_dev.csv & question_events.csv ({len(dev_df):,} rows, 1000 users)")

    # Test cohort (completely unseen)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_unseen_250.csv", index=False)
    print(f"  Saved: test_unseen_250.csv ({len(test_df):,} rows, 250 users)")

    # Lecture and explanation auxiliary tables
    if dev_l:
        l_df = pd.DataFrame(dev_l)
        l_df.to_csv(PROCESSED_DATA_DIR / "lecture_events.csv", index=False)
        print(f"  Saved: lecture_events.csv ({len(l_df):,} rows)")

    if dev_e:
        e_df = pd.DataFrame(dev_e)
        e_df.to_csv(PROCESSED_DATA_DIR / "explanation_events.csv", index=False)
        print(f"  Saved: explanation_events.csv ({len(e_df):,} rows)")

    # Metadata
    with open(PROCESSED_DATA_DIR / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Saved: metadata.pkl")

    questions_df.to_csv(PROCESSED_DATA_DIR / "questions_contents.csv", index=False)
    lectures_df.to_csv(PROCESSED_DATA_DIR / "lectures_contents.csv", index=False)
    print(f"  Saved: questions_contents.csv, lectures_contents.csv")

    print(f"\n  PREPROCESSING COMPLETE IN {time.time() - total_start:.1f}s!")
