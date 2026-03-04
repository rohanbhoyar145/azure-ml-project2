from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "FastAPI ML API"}
    
@app.post("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}