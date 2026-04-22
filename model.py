from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

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


# ── Tests ──────────────────────────────────────────────
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    response = client.post("/predict", json=[5.1, 3.5, 1.4, 0.2])
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["setosa", "versicolor", "virginica"]