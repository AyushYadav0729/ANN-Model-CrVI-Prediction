# ANN Model for Cr(VI) Removal Prediction

## Overview
This project develops an Artificial Neural Network (ANN) to predict chromium (Cr(VI)) removal efficiency using adsorption data.

## Features
- Input: Time, Concentration, pH, Dosage, Temperature, Adsorbent
- Output: Removal Efficiency (%)
- Model: ANN (128-64-32 architecture)
- Techniques: L2 Regularization, Dropout, MAE Loss

## Results
- R² Score = 8.27
- RMSE ≈ 8.7
- MAE ≈ 6.5

## Files
- train_ann_v3.py 
  Script used to train the ANN model.

- predict_ann_v3.py
  Script used to make predictions using the trained model.

- final_cleaned_dataset_v2.xlsx
  Dataset used for training.

## Requirements

Install these libraries before running the scripts:
```bash
pip install pandas numpy matplotlib tensorflow scikit-learn joblib openpyxl
```
## How to Run
- 1 ) Make sure these files are in one directory:
  - train_ann_v3.py
  - predict_ann_v3.py
  - final_cleaned_dataset_v2.xlsx

- 2 ) Train the model
  Open terminal / command prompt in that folder and run:
  ```bash
  python train_ann_v3.py
  ```
  this will save the trained model and preprocessing files and generate prediction and plot files

- 3 ) Run the prediction
  After training is complete
  Open predict_ann_v3.py and edit the values inside with your values:
  ```bash
  new_data = pd.DataFrame([{
    "Adsorbent": "WSB",
    "Time": 120,
    "Initial_Concentration": 110,
    "pH": 5.5,
    "Dosage": 1.1,
    "Temperature": 25
  }])
  ```

  then run:
  ```bash
  python predict_ann_v3.py
  ```

