import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# =========================
# 1. LOAD DATA
# =========================
file_path = "final_cleaned_dataset_v2.xlsx"
df = pd.read_excel(file_path)

print("First 5 rows of dataset:")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())
print("\nDataset shape:", df.shape)


# =========================
# 2. DEFINE INPUTS AND OUTPUT
# =========================
target_column = "Removal_Efficiency"

X = df.drop(columns=[target_column])
y = df[target_column].values


# =========================
# 3. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# =========================
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)


# =========================
# 4. PREPROCESSING
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

y = y.reshape(-1, 1)
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)


# =========================
# 5. TRAIN / VALIDATION / TEST SPLIT
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_processed, y_scaled, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("\nShapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)


# =========================
# 6. BUILD IMPROVED ANN MODEL
# =========================
input_dim = X_train.shape[1]

model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.10),

    Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.10),

    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss="mae",
    metrics=["mae"]
)

print("\nModel Summary:")
model.summary()


# =========================
# 7. CALLBACKS
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-5,
    verbose=1
)


# =========================
# 8. TRAIN MODEL
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# =========================
# 9. EVALUATE MODEL
# =========================
y_pred_scaled = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_actual = y_scaler.inverse_transform(y_test)

r2 = r2_score(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)

# avoid division by zero in MAPE
epsilon = 1e-8
mape = np.mean(np.abs((y_test_actual - y_pred) / (y_test_actual + epsilon))) * 100

print("\n===== TEST METRICS =====")
print(f"R²   = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"MAPE = {mape:.2f}%")


# =========================
# 10. SAVE PREDICTIONS
# =========================
results = pd.DataFrame({
    "Actual_Removal_Efficiency": y_test_actual.flatten(),
    "Predicted_Removal_Efficiency": y_pred.flatten(),
    "Error": y_test_actual.flatten() - y_pred.flatten()
})

results.to_excel("ANN_Predictions_v3.xlsx", index=False)
print("\nPredictions saved to ANN_Predictions_v3.xlsx")
print("\nPrediction sample:")
print(results.head())


# =========================
# 11. SAVE METRICS
# =========================
metrics_df = pd.DataFrame({
    "Metric": ["R2", "RMSE", "MAE", "MAPE"],
    "Value": [r2, rmse, mae, mape]
})
metrics_df.to_excel("ANN_Test_Metrics_v3.xlsx", index=False)
print("Metrics saved to ANN_Test_Metrics_v3.xlsx")


# =========================
# 12. SAVE MODEL AND PREPROCESSORS
# =========================
model.save("ann_removal_model_v3.keras")
joblib.dump(preprocessor, "preprocessor_v3.pkl")
joblib.dump(y_scaler, "y_scaler_v3.pkl")

print("Model saved as ann_removal_model_v3.keras")
print("Preprocessor saved as preprocessor_v3.pkl")
print("Output scaler saved as y_scaler_v3.pkl")


# =========================
# 13. PLOTS
# =========================

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve_v3.png")
plt.show()

# Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test_actual, y_pred, s=60)
min_val = min(y_test_actual.min(), y_pred.min())
max_val = max(y_test_actual.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")
plt.xlabel("Actual Removal Efficiency (%)")
plt.ylabel("Predicted Removal Efficiency (%)")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_v3.png")
plt.show()

# Error histogram
errors = y_test_actual.flatten() - y_pred.flatten()
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=15)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Histogram")
plt.grid(True)
plt.tight_layout()
plt.savefig("error_histogram_v3.png")
plt.show()

print("\nTraining completed successfully.")
