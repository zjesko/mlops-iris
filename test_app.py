from fastapi.testclient import TestClient
from main import app
from datetime import datetime

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong", "timestamp": datetime.now().strftime("%H:%M:%S")}


# test to check if Iris Virginica is classified correctly.
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 2.8,
        "sepal_width": 5,
        "petal_length": 3,
        "petal_width": 4.1,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica" , "timestamp": datetime.now().strftime("%H:%M:%S")}


    def test1():
        with TestClient(app) as client:
            response = client.post("/test1")
            # asserting the correct response is received
            assert response.status_code == 200
            assert response.json() == {"Test1": "Its just a test1", "timestamp": datetime.now().strftime("%H:%M:%S")}
        
    def test2():
        with TestClient(app) as client:
            response = client.post("/test2")
            # asserting the correct response is received
            assert response.status_code == 200
            assert response.json() == {"Test2": "Its just a test2", "timestamp": datetime.now().strftime("%H:%M:%S")}
