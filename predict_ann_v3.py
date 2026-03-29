import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load saved files
model = load_model("ann_removal_model_v3.keras")
preprocessor = joblib.load("preprocessor_v3.pkl")
y_scaler = joblib.load("y_scaler_v3.pkl")

# Example new sample
new_data = pd.DataFrame([{
    "Adsorbent": "AEBB",
    "Time": 100,
    "Initial_Concentration": 10,
    "pH": 3,
    "Dosage": 0.7,
    "Temperature": 25
}])

# Preprocess
X_new = preprocessor.transform(new_data)
if hasattr(X_new, "toarray"):
    X_new = X_new.toarray()

# Predict
y_pred_scaled = model.predict(X_new)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

print("\nInput sample:")
print(new_data)
print(f"\nPredicted Removal Efficiency = {y_pred[0][0]:.2f}%")
