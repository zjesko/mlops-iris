from fastapi.testclient import TestClient
from main import app
# from datatime import datetime

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
        # print(datatime.strftime("%H:%M:%S.%f",response.elapsed.total_seconds()))

# test to check if Iris Setosa is classified correctly
def test_pred_Setosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 4.6,
        "sepal_width": 3.1,
        "petal_length": 1.5,
        "petal_width": .2,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Setosa"}

# test to check if Iris Versicolor is classified correctly
def test_pred_Versicolour():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5.9,
        "sepal_width": 3.0,
        "petal_length": 4.2,
        "petal_width": 1.5,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Versicolour"}



