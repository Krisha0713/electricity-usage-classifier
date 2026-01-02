import numpy as np
import pandas as pd

np.random.seed(42)

def generate_pattern(label, length=24):
    if label == "normal":
        return np.random.normal(1.0, 0.05, length)
    elif label == "spike":
        data = np.random.normal(1.0, 0.05, length)
        data[np.random.randint(5, 20)] += np.random.uniform(0.8, 1.2)
        return data
    elif label == "rising":
        return np.linspace(0.8, 1.4, length) + np.random.normal(0, 0.03, length)

data, labels = [], []

for label in ["normal", "spike", "rising"]:
    for _ in range(100):
        data.append(generate_pattern(label))
        labels.append(label)

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv("electricity_usage.csv", index=False)

print("Dataset generated: electricity_usage.csv")
