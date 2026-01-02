import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from extract_features import extract_features

df = pd.read_csv("electricity_usage.csv")

X, y = [], []

for _, row in df.iterrows():
    series = row[:-1].values.astype(float)
    X.append(extract_features(series))
    y.append(row["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
