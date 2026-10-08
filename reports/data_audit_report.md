# Empirical Data Audit & Cohort Verification Report
**Audit Date:** 2026-09-12  
**Verification Scope:** Raw KT-3 (1000) Development Cohort & Raw KT-3 (250) TEST Cohort  
**Cohort Disjointness:** PASSED (Zero Overlap)

## 1. Cohort Ingestion & Entity Counts
| Cohort | Source Files | Extracted Students | Question Events | Mean Events/Student | Baseline Accuracy |
|---|---|---|---|---|---|
| **Development Cohort** | 998 | 998 | 532,594 | 533.7 | 67.05% |
| **Final Unseen Test** | 249 | 249 | 174,608 | 701.2 | 67.13% |

## 2. Vocabulary & Feature Mapping Bounds (Fitted Exclusively on Dev)
- **Unique Questions in Dev Vocabulary:** 11,335
- **Unique Concept Tags in Dev Vocabulary:** 189
- **TOEIC Parts Range:** 1 to 8
- **Response Time Scaler:** Mean = 38771.1 ms, Std = 47858.1 ms (Clip = 295236.4 ms)
- **Inter-Event Time Scaler:** Mean = 4221850.0 ms, Std = 21568994.0 ms (Clip = 169484140.3 ms)
- **Novel Questions in Test Set:** 86 interactions mapped safely to reserved index 0 (unknown/padding).

## 3. Data Integrity Verifications
- **Exact Duplicate Rows in Processed Dev:** 0
- **Target Leakage Check:** Target `is_correct` quarantined; `user_answer` and `correct_answer` excluded from input feature tensors.
- **Temporal Monotonicity:** Verified across all student sequences (no negative $\Delta t$ lookahead).