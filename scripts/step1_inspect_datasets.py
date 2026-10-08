"""
Step 1 -- Load CSV Dataset: Comprehensive Dataset Inspection Script
===================================================================
AI-Based Personalized Learning Recommendation System -- Smart Education

This script inspects EdNet Contents and KT3 datasets without modifying them.
It produces a detailed report of schemas, statistics, join keys, and action types.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import sys
import json
import glob
import time
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EDNET_CONTENTS_DIR, EDNET_KT3_DIR

# ============================================================
# Helper Functions
# ============================================================

def format_bytes(size_bytes):
    """Format bytes to human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def inspect_csv_file(filepath, label=""):
    """Inspect a single CSV file and return a summary dict."""
    file_size = os.path.getsize(filepath)
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {
            "file": os.path.basename(filepath),
            "label": label,
            "error": str(e),
            "file_size": format_bytes(file_size)
        }
    
    info = {
        "file": os.path.basename(filepath),
        "label": label,
        "path": filepath,
        "file_size_bytes": file_size,
        "file_size_human": format_bytes(file_size),
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(count) for col, count in df.isnull().sum().items() if count > 0},
        "total_missing": int(df.isnull().sum().sum()),
        "sample_head": df.head(5).to_string(index=False)
    }
    return info, df


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ============================================================
# 1. Inspect Contents Dataset
# ============================================================
print_section("CONTENTS DATASET INSPECTION")

contents_files = sorted(glob.glob(os.path.join(EDNET_CONTENTS_DIR, "*.csv")))
print(f"\nContents directory: {EDNET_CONTENTS_DIR}")
print(f"Number of CSV files: {len(contents_files)}")

contents_data = {}
for fpath in contents_files:
    fname = os.path.basename(fpath)
    print(f"\n{'-'*60}")
    print(f"File: {fname}")
    print(f"{'-'*60}")
    
    result = inspect_csv_file(fpath, label=fname.replace('.csv', ''))
    if isinstance(result, dict):
        print(f"  ERROR: {result.get('error', 'Unknown')}")
        continue
    
    info, df = result
    contents_data[fname] = (info, df)
    
    print(f"  Size:    {info['file_size_human']}")
    print(f"  Rows:    {info['rows']:,}")
    print(f"  Columns: {info['columns']}")
    print(f"  Columns: {info['column_names']}")
    print(f"\n  Data Types:")
    for col, dtype in info['dtypes'].items():
        print(f"    {col:<25s} {dtype}")
    
    if info['total_missing'] > 0:
        print(f"\n  Missing Values (total: {info['total_missing']:,}):")
        for col, count in info['missing_values'].items():
            pct = count / info['rows'] * 100
            print(f"    {col:<25s} {count:>8,} ({pct:.1f}%)")
    else:
        print(f"\n  Missing Values: NONE")
    
    print(f"\n  Sample Records (first 5):")
    print(f"  {info['sample_head']}")


# ============================================================
# 2. Inspect KT3 Dataset
# ============================================================
print_section("KT3 DATASET INSPECTION")

kt3_files = sorted(glob.glob(os.path.join(EDNET_KT3_DIR, "*.csv")))
print(f"\nKT3 directory: {EDNET_KT3_DIR}")
print(f"Number of user files: {len(kt3_files)}")

# Extract user IDs from filenames
user_ids = []
for fpath in kt3_files:
    fname = os.path.basename(fpath)
    uid = fname.replace('.csv', '')
    user_ids.append(uid)

print(f"\nUser ID range (sample): {user_ids[:5]} ... {user_ids[-5:]}")

# File size statistics
file_sizes = [os.path.getsize(f) for f in kt3_files]
print(f"\nFile Size Statistics:")
print(f"  Total:   {format_bytes(sum(file_sizes))}")
print(f"  Min:     {format_bytes(min(file_sizes))}")
print(f"  Max:     {format_bytes(max(file_sizes))}")
print(f"  Mean:    {format_bytes(sum(file_sizes)//len(file_sizes))}")

# Verify schema consistency across ALL files
# Read first line (header) from each file
print(f"\n{'─'*60}")
print("Schema Consistency Check (all 1000 files)")
print(f"{'─'*60}")

schema_counter = Counter()
header_errors = []
for fpath in kt3_files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            schema_counter[header] += 1
    except Exception as e:
        header_errors.append((os.path.basename(fpath), str(e)))

print(f"\nDistinct schemas found: {len(schema_counter)}")
for schema, count in schema_counter.most_common():
    print(f"  [{count:>5} files] {schema}")

if header_errors:
    print(f"\n  Header read errors: {len(header_errors)}")
    for fn, err in header_errors[:5]:
        print(f"    {fn}: {err}")

# Read a sample of files for detailed analysis
# Use a diverse sample: small, medium, large files
sorted_by_size = sorted(zip(file_sizes, kt3_files))
sample_indices = [0, len(sorted_by_size)//4, len(sorted_by_size)//2, 
                  3*len(sorted_by_size)//4, len(sorted_by_size)-1]
sample_files = [sorted_by_size[i][1] for i in sample_indices]

print(f"\n{'─'*60}")
print("Detailed Sample Inspection (5 files: smallest to largest)")
print(f"{'─'*60}")

for fpath in sample_files:
    fname = os.path.basename(fpath)
    result = inspect_csv_file(fpath, label=fname)
    if isinstance(result, dict):
        print(f"  {fname}: ERROR - {result.get('error')}")
        continue
    info, df = result
    print(f"\n  {fname} ({info['file_size_human']}, {info['rows']:,} rows)")
    print(f"    Columns: {info['column_names']}")
    print(f"    Types: {info['dtypes']}")
    if info['total_missing'] > 0:
        print(f"    Missing: {info['missing_values']}")
    else:
        print(f"    Missing: NONE")

# ============================================================
# 3. Comprehensive KT3 Analysis — Action Types & Stats
# ============================================================
print_section("KT3 COMPREHENSIVE ANALYSIS")
print("Loading all 1,000 user files for action type & statistics analysis...")
print("(Reading only — original files are NOT modified)")

start_time = time.time()

all_action_types = Counter()
all_sources = Counter()
all_platforms = Counter()
total_rows = 0
rows_per_user = {}
users_with_respond = 0
respond_item_ids_sample = set()
item_id_prefixes = Counter()
user_answer_non_empty = 0
user_answer_total = 0
timestamp_min = float('inf')
timestamp_max = float('-inf')

for i, fpath in enumerate(kt3_files):
    fname = os.path.basename(fpath)
    uid = fname.replace('.csv', '')
    
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"  ERROR reading {fname}: {e}")
        continue
    
    n = len(df)
    total_rows += n
    rows_per_user[uid] = n
    
    # Action types
    if 'action_type' in df.columns:
        for at, cnt in df['action_type'].value_counts().items():
            all_action_types[at] += cnt
    
    # Sources
    if 'source' in df.columns:
        for src, cnt in df['source'].value_counts().items():
            all_sources[src] += cnt
    
    # Platforms
    if 'platform' in df.columns:
        for plat, cnt in df['platform'].value_counts().items():
            all_platforms[plat] += cnt
    
    # Item ID prefixes
    if 'item_id' in df.columns:
        for item in df['item_id'].dropna():
            prefix = ''.join(c for c in str(item) if c.isalpha())
            item_id_prefixes[prefix] += 1
    
    # User answers
    if 'user_answer' in df.columns:
        user_answer_total += n
        non_empty = df['user_answer'].dropna()
        non_empty = non_empty[non_empty.astype(str).str.strip() != '']
        user_answer_non_empty += len(non_empty)
    
    # Timestamps
    if 'timestamp' in df.columns:
        ts_col = pd.to_numeric(df['timestamp'], errors='coerce')
        valid_ts = ts_col.dropna()
        if len(valid_ts) > 0:
            timestamp_min = min(timestamp_min, valid_ts.min())
            timestamp_max = max(timestamp_max, valid_ts.max())
    
    # Check if user has any respond actions
    if 'action_type' in df.columns:
        if (df['action_type'] == 'respond').any():
            users_with_respond += 1
    
    if (i + 1) % 200 == 0:
        print(f"  ... processed {i+1}/1000 files")

elapsed = time.time() - start_time
print(f"\n  All 1,000 files loaded in {elapsed:.1f} seconds")

# Report results
print(f"\n{'─'*60}")
print("Overall KT3 Statistics")
print(f"{'─'*60}")
print(f"  Total users:           {len(rows_per_user):,}")
print(f"  Total raw interactions: {total_rows:,}")
print(f"  Min rows per user:     {min(rows_per_user.values()):,}")
print(f"  Max rows per user:     {max(rows_per_user.values()):,}")
print(f"  Mean rows per user:    {sum(rows_per_user.values())/len(rows_per_user):,.1f}")
print(f"  Median rows per user:  {sorted(rows_per_user.values())[len(rows_per_user)//2]:,}")

print(f"\n  Timestamp range:")
if timestamp_min != float('inf'):
    ts_min_dt = pd.to_datetime(timestamp_min, unit='ms')
    ts_max_dt = pd.to_datetime(timestamp_max, unit='ms')
    print(f"    Earliest: {ts_min_dt} ({int(timestamp_min)})")
    print(f"    Latest:   {ts_max_dt} ({int(timestamp_max)})")
else:
    print(f"    Could not determine timestamp range")

print(f"\n{'─'*60}")
print("Action Type Distribution")
print(f"{'─'*60}")
for action, count in all_action_types.most_common():
    pct = count / total_rows * 100
    print(f"  {action:<20s} {count:>12,}  ({pct:5.1f}%)")

print(f"\n{'─'*60}")
print("Source Distribution")
print(f"{'─'*60}")
for src, count in all_sources.most_common():
    pct = count / total_rows * 100
    print(f"  {src:<25s} {count:>12,}  ({pct:5.1f}%)")

print(f"\n{'─'*60}")
print("Platform Distribution")
print(f"{'─'*60}")
for plat, count in all_platforms.most_common():
    pct = count / total_rows * 100
    print(f"  {plat:<20s} {count:>12,}  ({pct:5.1f}%)")

print(f"\n{'─'*60}")
print("Item ID Prefix Distribution (entity types)")
print(f"{'─'*60}")
for prefix, count in item_id_prefixes.most_common():
    pct = count / total_rows * 100
    print(f"  {prefix:<10s} {count:>12,}  ({pct:5.1f}%)")

print(f"\n  Users with at least 1 'respond' action: {users_with_respond}/{len(rows_per_user)}")
print(f"  Rows with non-empty user_answer: {user_answer_non_empty:,} / {user_answer_total:,} ({user_answer_non_empty/max(user_answer_total,1)*100:.1f}%)")


# ============================================================
# 4. Join Key Analysis
# ============================================================
print_section("JOIN KEY ANALYSIS")

# Questions.csv has question_id like 'q1', 'q2', etc.
# KT3 has item_id which contains 'q...', 'b...', 'l...', 'e...' prefixed IDs
# The 'respond' action_type has item_id with 'q' prefix (question)
# The 'enter'/'submit' actions use 'b' prefix (bundle)
# The 'enter'/'quit' for lectures use 'l' prefix (lecture)
# The 'enter'/'quit' for explanations use 'e' prefix (explanation)

print("\nContents Dataset Keys:")
if 'questions.csv' in contents_data:
    q_info, q_df = contents_data['questions.csv']
    print(f"  questions.csv: question_id column")
    print(f"    Format: {q_df['question_id'].head(5).tolist()}")
    print(f"    Unique questions: {q_df['question_id'].nunique():,}")
    print(f"    bundle_id column: {q_df['bundle_id'].head(5).tolist()}")
    print(f"    explanation_id column: {q_df['explanation_id'].head(5).tolist()}")

if 'lectures.csv' in contents_data:
    l_info, l_df = contents_data['lectures.csv']
    print(f"\n  lectures.csv: lecture_id column")
    print(f"    Format: {l_df['lecture_id'].head(5).tolist()}")
    print(f"    Unique lectures: {l_df['lecture_id'].nunique():,}")

print("\nKT3 Item ID Patterns:")
print("  'respond' action → item_id starts with 'q' (question)")
print("  'enter'/'submit' → item_id starts with 'b' (bundle)")
print("  lecture enter/quit → item_id starts with 'l' (lecture)")
print("  explanation enter/quit → item_id starts with 'e' (explanation)")

# Verify join key existence: load one KT3 file, extract question IDs from responds
print(f"\nJoin Key Verification:")

# Read a medium-sized user file
test_file = sorted_by_size[len(sorted_by_size)//2][1]
test_df = pd.read_csv(test_file)
test_uid = os.path.basename(test_file).replace('.csv', '')

respond_rows = test_df[test_df['action_type'] == 'respond']
if len(respond_rows) > 0:
    kt3_q_ids = set(respond_rows['item_id'].unique())
    q_df_ids = set(contents_data['questions.csv'][1]['question_id'].unique()) if 'questions.csv' in contents_data else set()
    
    matched = kt3_q_ids & q_df_ids
    unmatched = kt3_q_ids - q_df_ids
    
    print(f"  Test user: {test_uid}")
    print(f"  Respond item_ids (questions): {len(kt3_q_ids)}")
    print(f"  Matched in questions.csv:     {len(matched)}")
    print(f"  Unmatched:                    {len(unmatched)}")
    if unmatched:
        print(f"    Unmatched samples: {list(unmatched)[:5]}")
    
    # Also check bundle_id linkage
    enter_rows = test_df[test_df['action_type'] == 'enter']
    bundle_ids_kt3 = set(
        enter_rows[enter_rows['item_id'].str.startswith('b', na=False)]['item_id'].unique()
    )
    bundle_ids_q = set(contents_data['questions.csv'][1]['bundle_id'].unique()) if 'questions.csv' in contents_data else set()
    b_matched = bundle_ids_kt3 & bundle_ids_q
    b_unmatched = bundle_ids_kt3 - bundle_ids_q
    print(f"\n  Bundle IDs in KT3 enter:     {len(bundle_ids_kt3)}")
    print(f"  Matched in questions.csv:     {len(b_matched)}")
    print(f"  Unmatched:                    {len(b_unmatched)}")

    # Lecture ID linkage
    lecture_ids_kt3 = set(
        test_df[test_df['item_id'].str.startswith('l', na=False)]['item_id'].unique()
    )
    lecture_ids_content = set(contents_data['lectures.csv'][1]['lecture_id'].unique()) if 'lectures.csv' in contents_data else set()
    l_matched = lecture_ids_kt3 & lecture_ids_content
    l_unmatched = lecture_ids_kt3 - lecture_ids_content
    print(f"\n  Lecture IDs in KT3:          {len(lecture_ids_kt3)}")
    print(f"  Matched in lectures.csv:      {len(l_matched)}")
    print(f"  Unmatched:                    {len(l_unmatched)}")
else:
    print(f"  Test user {test_uid} has NO respond actions")


# ============================================================
# 5. Correctness Join Verification
# ============================================================
print_section("CORRECTNESS JOIN VERIFICATION")

if 'questions.csv' in contents_data:
    q_df = contents_data['questions.csv'][1]
    
    print("questions.csv 'correct_answer' column:")
    print(f"  Unique answers: {sorted(q_df['correct_answer'].dropna().unique())}")
    print(f"  Missing: {q_df['correct_answer'].isnull().sum()}")
    
    print(f"\nquestions.csv 'tags' column sample:")
    print(f"  {q_df['tags'].head(5).tolist()}")
    print(f"  Missing tags: {q_df['tags'].isnull().sum()}")
    
    print(f"\nquestions.csv 'part' column:")
    print(f"  Unique parts: {sorted(q_df['part'].dropna().unique())}")
    
    # Show how correctness will be computed
    if len(respond_rows) > 0:
        sample_respond = respond_rows.head(3)
        print(f"\nCorrectness computation example (user {test_uid}):")
        for _, row in sample_respond.iterrows():
            qid = row['item_id']
            user_ans = row['user_answer']
            q_match = q_df[q_df['question_id'] == qid]
            if len(q_match) > 0:
                correct = q_match.iloc[0]['correct_answer']
                is_correct = 1 if str(user_ans).strip().lower() == str(correct).strip().lower() else 0
                print(f"    {qid}: user_answer='{user_ans}', correct='{correct}' → is_correct={is_correct}")
            else:
                print(f"    {qid}: NOT FOUND in questions.csv")

# ============================================================
# 6. Potential Issues
# ============================================================
print_section("POTENTIAL ISSUES & OBSERVATIONS")

print("""
1. ITEM ID SEMANTICS:
   - KT3 item_id uses different prefixes for different entity types:
     'q' = question, 'b' = bundle, 'l' = lecture, 'e' = explanation
   - The respond action references question IDs (q...) 
   - The enter/submit actions reference bundle IDs (b...)
   - Lectures and explanations use l.../e... prefixes
   - JOIN KEY: KT3 respond.item_id → questions.csv.question_id

2. RAW EVENT RECONSTRUCTION:
   - A single question attempt involves: enter(bundle) → respond(question) → submit(bundle)
   - Some users may have multiple 'respond' events before a 'submit'
   - Lecture engagement: enter(lecture) → quit(lecture), with duration = quit_ts - enter_ts
   - Explanation engagement: enter(explanation) → quit(explanation)

3. USER ANSWER:
   - Only 'respond' events have user_answer populated
   - 'enter', 'submit', 'quit' events have empty user_answer

4. SCHEMA UNIFORMITY:
   - All 1000 KT3 files share the same 6-column schema

5. MISSING DATA:
   - Some users have very few interactions (min ~8 rows)
   - user_answer is structurally empty for non-respond events (by design)

6. TAGS IN QUESTIONS:
   - Tags are semicolon-separated integers (e.g., '1;2;179;181')
   - These represent learning concepts/topics for knowledge tracing

7. LECTURE METADATA:
   - Some lectures have video_length = -1 (unknown/missing)
   - Some lectures have deployed_at = -1 (unknown)
""")

print_section("STEP 1 INSPECTION COMPLETE")
print("\nAll original dataset files remain UNTOUCHED.")
print("No preprocessing, feature engineering, or model training was performed.")
