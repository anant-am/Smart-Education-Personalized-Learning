"""
STANDARDIZED MAPPING: BERT-style Transformer + Neural Collaborative Filtering (MODEL 3)
========================================================================================
In the university standardized numbering:
  - MODEL 3: BERT-style Transformer + Neural Collaborative Filtering (scripts/run_model3.py)
  - MODEL 4: Autoencoder + Recommender Network (scripts/run_model4.py)

For full backward compatibility with legacy execution references,
this script executes the standardized Model 3 BERT-NCF lifecycle:
1. Load verified Development cohort splits (train.csv, val.csv) and isolated Unseen Test (test.csv).
2. Training-Only SMOTE Verification & Isolation Audit.
3. 10-Fold Grouped Cross-Validation grouped by actual student user_id.
4. Initial Model Training (Transformer sequence encoder + NeuMF item scoring).
5. Quantitative Overfitting/Underfitting Diagnosis with empirical evidence.
6. Retraining Invariant (fresh ModelTrainer & new optimizer).
7. Final Untouched Test Evaluation on 250 unseen students.
8. Dual Knowledge State Estimation (Model-Derived vs. Historical Baseline), Gap Detection, and Recommendation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.run_model3 as run_m3

def main():
    print("[STANDARDIZED NUMBERING NOTICE]")
    print("BERT-style Transformer + NCF is MODEL 3 in the university standardized specification.")
    print("Executing standardized Model 3 lifecycle via scripts/run_model3.py...\n")
    return run_m3.main()

if __name__ == "__main__":
    main()
