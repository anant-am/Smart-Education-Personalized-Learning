# Dataset Splitting & Provenance Report
**Split Generation Timestamp:** 2026-10-12 13:28:42  
**Random Seed:** 42  
**Split Protocol:** Strict User-Level 80:20 Split across full cohort; Zero Cross-Cohort Overlap

## 1. Partition Breakdown
| Partition | Source Cohort | Split Strategy | Student Count | Interaction Events | Class 0 (Incorrect) | Class 1 (Correct) | Baseline Accuracy |
|---|---|---|---|---|---|---|---|
| **Training (`train.csv`)** | 80% Dev Cohort | User-Level Partition | 189,861 | 11,450,396 | 4,027,186 | 7,423,210 | 64.83% |
| **Validation (`val.csv`)** | 20% Dev Cohort | User-Level Partition | 47,486 | 2,894,347 | 1,006,608 | 1,887,739 | 65.22% |
| **Frozen Test (`test.csv`)** | 20% Test Cohort | 100% Unseen Students | 59,353 | 3,556,119 | 1,257,338 | 2,298,781 | 64.64% |

## 2. Partition Isolation Guarantees
- **Cohort Isolation:** $\text{Dev} \cap \text{Test} = \emptyset$. The 20% test students are completely isolated and frozen.
- **Zero Augmentation of Test/Val:** The validation and test sets remain in their natural un-augmented distribution.
- **Preprocessing Bounds:** Normalization and vocabularies fitted exclusively on Training students; Test cohort transformed using frozen parameters.