import numpy as np

def extract_features(signal):
    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "max": np.max(signal),
        "min": np.min(signal)
    }
    return features