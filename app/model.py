from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = FastAPI()

# Train a simple model on startup
iris = load_iris()
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(iris.data, iris.target)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: list[float]):
    prediction = clf.predict([features])
    label = iris.target_names[prediction[0]]
    return {"prediction": label}