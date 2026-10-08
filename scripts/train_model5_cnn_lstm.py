"""
MODEL 5: CNN + LSTM Hybrid Architecture
=======================================
University Standardized Model 5:
Executes the verified academic lifecycle for Model 5 (CNN + LSTM):
1. Load verified Development cohort splits (train.csv, val.csv) and isolated Unseen Test (test.csv).
2. Training-Only SMOTE Verification & Isolation Audit.
3. 10-Fold Grouped Cross-Validation grouped by actual student user_id.
4. Initial Model Training (1D Temporal Convolution + LSTM sequence modeling).
5. Quantitative Overfitting/Underfitting Diagnosis with empirical evidence.
6. Correction Technique Application.
7. Retraining Invariant (fresh ModelTrainer & new optimizer).
8. Final Untouched Test Evaluation on 250 unseen students.
9. Dual Knowledge State Estimation (Model-Derived vs. Historical Baseline), Gap Detection, and Recommendation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.run_model5 as run_m5

def main():
    return run_m5.main()

if __name__ == "__main__":
    main()
