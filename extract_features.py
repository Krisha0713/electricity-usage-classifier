import numpy as np

def extract_features(series):
    return [
        np.mean(series),
        np.std(series),
        np.max(series) - np.min(series),
        np.polyfit(range(len(series)), series, 1)[0],  # trend slope
        np.sum(series ** 2)  # energy
    ]
