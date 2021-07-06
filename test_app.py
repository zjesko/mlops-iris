from fastapi.testclient import TestClient
from main import app


def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"ping":"pong"}

def test_pred_virginica():
    payload = {
      "sepal_length": 3,
      "sepal_width": 5,
      "petal_length": 3.2,
      "petal_width": 4.4
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        assert response.status_code == 200
        assert response.json() == {'flower_class': "Iris Virginica"}


def test_pred_versicolor():
    payload = {
      "sepal_length": 5.8,
      "sepal_width": 2.7,
      "petal_length": 3.9,
      "petal_width": 1.2
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        assert response.status_code == 200
        assert response.json() == {'flower_class': "Iris Versicolour"}


def test_pred_setosa():
    payload = {
      "sepal_length": 4.5,
      "sepal_width": 2.3,
      "petal_length": 1.3,
      "petal_width": 0.3
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        assert response.status_code == 200
        assert response.json() == {'flower_class': "Iris Setosa"}