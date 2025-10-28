import pandas as pd
import numpy as np
from asl_official import ASLOfficialClassifier

# Create simple test data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.random.choice([0, 1], size=1000, p=[0.965, 0.035])  # Mimic GSS imbalance

print(f"Test data: {len(X)} samples, {np.sum(y)} positive class ({np.mean(y):.3f})")

try:
    print("Testing ASL Official Classifier...")
    asl_model = ASLOfficialClassifier(gamma_neg=4, gamma_pos=1, epochs=5, batch_size=64)
    asl_model.fit(X, y)
    probs = asl_model.predict_proba(X)
    print(f"ASL training successful! Predictions shape: {probs.shape}")
    print(f"Positive class probability range: {probs[:, 1].min():.4f} - {probs[:, 1].max():.4f}")
except Exception as e:
    print(f"ASL training failed: {e}")
    import traceback
    traceback.print_exc()