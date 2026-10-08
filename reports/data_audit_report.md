# Empirical Data Audit & Cohort Verification Report
**Audit Date:** 2026-10-08  
**Verification Scope:** Raw EdNet-KT3 User-Level 80:20 Split  
**Cohort Disjointness:** PASSED (Zero Overlap)

## 1. Cohort Ingestion & Entity Counts
| Cohort | Source Files | Extracted Students | Question Events | Mean Events/Student | Baseline Accuracy |
|---|---|---|---|---|---|
| **Development Cohort** | 800 | 799 | 524,807 | 656.8 | 68.50% |
| **Final Frozen Test** | 200 | 199 | 159,356 | 800.8 | 69.75% |

## 2. Vocabulary & Feature Mapping Bounds (Fitted Exclusively on Dev)
- **Unique Questions in Dev Vocabulary:** 11,528
- **Unique Concept Tags in Dev Vocabulary:** 187
- **TOEIC Parts Range:** 1 to 7
- **Response Time Scaler:** Mean = 46659.5 ms, Std = 53330.2 ms (Clip = 309507.5 ms)
- **Inter-Event Time Scaler:** Mean = 3190510.8 ms, Std = 17596371.7 ms (Clip = 142492761.4 ms)
- **Novel Questions in Test Set:** 23 interactions mapped safely to reserved index 0 (unknown/padding).

## 3. Data Integrity Verifications
- **Exact Duplicate Rows in Processed Dev:** 0
- **Target Leakage Check:** Target `is_correct` quarantined; `user_answer` and `correct_answer` excluded from input feature tensors.
- **Temporal Monotonicity:** Verified across student sequences (no negative delta-t lookahead).