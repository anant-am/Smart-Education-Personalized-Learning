"""
STANDARDIZED MAPPING: Autoencoder + Recommender Network (MODEL 4)
==================================================================
In the university standardized numbering:
  - MODEL 2: Transformer + Knowledge Tracing (scripts/run_model2.py)
  - MODEL 4: Autoencoder + Recommender Network (scripts/run_model4.py)

For full backward compatibility with legacy execution references,
this script executes the standardized Model 4 Autoencoder lifecycle:
1. Build interaction profiles from Development cohort (1000 users) and Unseen Test (250 users).
2. Resource vocabulary fitted exclusively on Development cohort.
3. 10-Fold Cross-Validation on Development interaction profiles.
4. Initial Model Training & Overfitting Diagnosis.
5. Retraining Invariant (fresh ModelTrainer & new optimizer).
6. Recommendation Ranking Metrics on Unseen Test cohort (Precision@K, Recall@K, NDCG@K, HitRate@K).
7. Dual Knowledge State & Personalized Recommendation Demo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.run_model4 as run_m4

def main():
    print("[STANDARDIZED NUMBERING NOTICE]")
    print("Autoencoder + Recommender Network is MODEL 4 in the university standardized specification.")
    print("Executing standardized Model 4 lifecycle via scripts/run_model4.py...\n")
    return run_m4.main()

if __name__ == "__main__":
    main()
