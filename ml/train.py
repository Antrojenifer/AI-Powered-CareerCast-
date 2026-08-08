"""
Milestone 1 baseline trainer (Logistic Regression + TF-IDF).
For Milestone 2 advanced training use: python -m ml.train_advanced
"""
from ml.train_advanced import train_all

if __name__ == "__main__":
    # Milestone 1 compatible: TF-IDF only, still trains RF + XGB
    train_all(use_sbert=False)
