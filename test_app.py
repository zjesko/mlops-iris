from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica"}

# task 2

# test to check if Iris Setosa is classified correctly
def test_pred_setosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Setosa"}

# test to check if Iris Versicolour is classified correctly
def test_pred_versicolour():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5.4,
        "petal_length": 4.5,
        "petal_width": 1.5,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Versicolour"}