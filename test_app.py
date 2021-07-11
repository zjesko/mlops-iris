from fastapi.testclient import TestClient
from main import app
from datetime import datetime

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong", "TimeStamp": datetime.now().strftime("%m, %d, %Y")}


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

#Test cases added my Gopi

def test_gopi1():
    #defining sample payload for test case
    payload = {
        "sepal_length": 0,
        "sepal_width": 0,
        "petal_length": 1,
        "petal_width": 0
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower",json=payload)
        # confirming correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Setosa"}

def test_gopi2():
    #defining sample payload for test case
    payload = {
        "sepal_length": 3,
        "sepal_width": 3,
        "petal_length": 2,
        "petal_width": 1
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower",json=payload)
        # confirming correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Versicolour"}
