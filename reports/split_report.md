# Dataset Splitting & Provenance Report
**Split Generation Timestamp:** 2026-10-08 22:55:45  
**Random Seed:** 42  
**Split Protocol:** Strict User-Level 80:20 Split across full cohort; Zero Cross-Cohort Overlap

## 1. Partition Breakdown
| Partition | Source Cohort | Split Strategy | Student Count | Interaction Events | Class 0 (Incorrect) | Class 1 (Correct) | Baseline Accuracy |
|---|---|---|---|---|---|---|---|
| **Training (`train.csv`)** | 80% Dev Cohort | User-Level Partition | 639 | 396,290 | 124,998 | 271,292 | 68.46% |
| **Validation (`val.csv`)** | 20% Dev Cohort | User-Level Partition | 160 | 128,517 | 40,311 | 88,206 | 68.63% |
| **Frozen Test (`test.csv`)** | 20% Test Cohort | 100% Unseen Students | 199 | 159,356 | 48,201 | 111,155 | 69.75% |

## 2. Partition Isolation Guarantees
- **Cohort Isolation:** $\text{Dev} \cap \text{Test} = \emptyset$. The 20% test students are completely isolated and frozen.
- **Zero Augmentation of Test/Val:** The validation and test sets remain in their natural un-augmented distribution.
- **Preprocessing Bounds:** Normalization and vocabularies fitted exclusively on Training students; Test cohort transformed using frozen parameters.