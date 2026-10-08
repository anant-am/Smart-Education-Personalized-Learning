"""
STANDARDIZED MAPPING: Transformer + Knowledge Tracing (MODEL 2)
================================================================
In the university standardized numbering:
  - MODEL 2: Transformer + Knowledge Tracing (scripts/run_model2.py)
  - MODEL 3: BERT-style Transformer + Neural Collaborative Filtering (scripts/run_model3.py)

For full backward compatibility with legacy execution references,
this script executes the standardized Model 2 Transformer lifecycle:
1. Load verified Development cohort splits (train.csv, val.csv) and isolated Unseen Test (test.csv).
2. Training-Only SMOTE Verification & Isolation Audit.
3. 10-Fold Grouped Cross-Validation grouped by actual student user_id.
4. Initial Model Training with causal self-attention masking and checkpointing.
5. Quantitative Overfitting/Underfitting Diagnosis with empirical evidence.
6. Retraining Invariant (fresh ModelTrainer & new optimizer).
7. Final Untouched Test Evaluation on 250 unseen students.
8. Dual Knowledge State Estimation (Model-Derived vs. Historical Baseline), Gap Detection, and Recommendation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.run_model2 as run_m2

def main():
    print("[STANDARDIZED NUMBERING NOTICE]")
    print("Transformer + Knowledge Tracing is MODEL 2 in the university standardized specification.")
    print("Executing standardized Model 2 lifecycle via scripts/run_model2.py...\n")
    return run_m2.main()

if __name__ == "__main__":
    main()
