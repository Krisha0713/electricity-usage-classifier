import numpy as np
import joblib
from extract_features import extract_features

model = joblib.load("model.pkl")

sample = np.random.normal(1.0, 0.05, 24)
features = extract_features(sample)

prediction = model.predict([features])
print("Predicted usage pattern:", prediction[0])
