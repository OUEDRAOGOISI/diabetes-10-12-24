from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# Charger le modèle et le scaler
print("Chargement du modèle...")
model = joblib.load("best_random_forest_model.pkl")
print("Modèle chargé avec succès.")

print("Chargement du scaler...")
scaler = joblib.load("scaler.pkl")
print("Scaler chargé avec succès.")


# Initialiser l'application FastAPI
app = FastAPI(
    title="Diabetes Prediction API",
    description="An API to predict diabetes using a trained model",
    version="1.0.0",
    openapi_url="/openapi.json"
)

# Définir un schéma pour les requêtes
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(input_data: DiabetesInput):
    # Convertir les données en tableau numpy
    data = np.array([[input_data.Pregnancies, input_data.Glucose, input_data.BloodPressure,
                      input_data.SkinThickness, input_data.Insulin, input_data.BMI,
                      input_data.DiabetesPedigreeFunction, input_data.Age]])
    # Normaliser les données
    data_scaled = scaler.transform(data)
    # Faire une prédiction
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0][1]

    return {"prediction": int(prediction[0]), "probability": float(probability)}

@app.get("/")
def root():
    return {"message": "Welcome to the Diabetes Prediction API! Visit /docs for more information."}
