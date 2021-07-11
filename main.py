import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model, predict, retrain
from typing import List
from datetime import datetime

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S")

# defining the main app
app = FastAPI(title="Iris Predictor", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", load_model)

# class which is expected in the payload
class QueryIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# class which is returned in the response
class QueryOut(BaseModel):
    flower_class: str

# class which is expected in the payload while re-training
class FeedbackIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    flower_class: str

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/predict_flower", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def predict_flower(query_data: QueryIn):
    output = {"flower_class": predict(query_data), "date stamp": date}
    return output

@app.post("/feedback_loop", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct flower class
# Response: Dict with detail confirming success (200)
def feedback_loop(data: List[FeedbackIn]):
    retrain(data)
    return {"detail": "Feedback loop successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
