# Empirical Data Audit & Cohort Verification Report
**Audit Date:** 2026-10-12  
**Verification Scope:** Raw EdNet-KT3 User-Level 80:20 Split  
**Cohort Disjointness:** PASSED (Zero Overlap)

## 1. Cohort Ingestion & Entity Counts
| Cohort | Source Files | Extracted Students | Question Events | Mean Events/Student | Baseline Accuracy |
|---|---|---|---|---|---|
| **Development Cohort** | 238,332 | 237,347 | 14,344,743 | 60.4 | 64.91% |
| **Final Frozen Test** | 59,583 | 59,353 | 3,556,119 | 59.9 | 64.64% |

## 2. Vocabulary & Feature Mapping Bounds (Fitted Exclusively on Dev)
- **Unique Questions in Dev Vocabulary:** 11,555
- **Unique Concept Tags in Dev Vocabulary:** 187
- **TOEIC Parts Range:** 1 to 7
- **Response Time Scaler:** Mean = 38057.2 ms, Std = 45269.5 ms (Clip = 281380.0 ms)
- **Inter-Event Time Scaler:** Mean = 4043923.0 ms, Std = 22621588.0 ms (Clip = 181645968.0 ms)
- **Novel Questions in Test Set:** 0 interactions mapped safely to reserved index 0 (unknown/padding).

## 3. Data Integrity Verifications
- **Exact Duplicate Rows in Processed Dev:** 0
- **Target Leakage Check:** Target `is_correct` quarantined; `user_answer` and `correct_answer` excluded from input feature tensors.
- **Temporal Monotonicity:** Verified across student sequences (no negative delta-t lookahead).