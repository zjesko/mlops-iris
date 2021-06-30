import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model, predict

# defining the main app
app = FastAPI(title="Iris Predictor", docs_url="/")

# calling the load_model during startup
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


# Route definitions
@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict_flower", response_model=QueryOut, status_code=200)
def predict_flower(query_data: QueryIn):
    output = {"flower_class": predict(query_data)}
    return output


# Main function to start the app when main.py is called
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
