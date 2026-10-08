# Dataset Splitting & Provenance Report
**Split Generation Timestamp:** 2026-09-12 15:25:47  
**Random Seed:** 42  
**Split Protocol:** Chronological 80/20 Per-Student Split on Development Cohort + Frozen Unseen Test Cohort

## 1. Partition Breakdown
| Partition | Source Cohort | Split Strategy | Student Count | Interaction Events | Class 0 (Incorrect) | Class 1 (Correct) | Baseline Accuracy |
|---|---|---|---|---|---|---|---|
| **Training (`train.csv`)** | KT-3 (1000) | First 80% chronological events | 998 | 425,675 | 141,257 | 284,418 | 66.82% |
| **Validation (`val.csv`)** | KT-3 (1000) | Final 20% chronological events | 993 | 106,919 | 34,237 | 72,682 | 67.98% |
| **Test (`test.csv`)** | KT-3 (250) TEST | 100% Unseen Students | 249 | 174,608 | 57,398 | 117,210 | 67.13% |

## 2. Partition Isolation Guarantees
- **Cohort Isolation:** $\text{Dev} \cap \text{Test} = \emptyset$. The 250 test students are completely isolated and untouched.
- **Temporal Integrity:** In `train.csv` and `val.csv`, every validation interaction occurs chronologically strictly *after* that same student's training interactions.
- **Preprocessing Bounds:** Normalization and vocabularies fitted exclusively on the 1,000 Development students; Test cohort transformed using frozen parameters.