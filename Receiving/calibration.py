import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

# ===========================
# User input
# ===========================
filename = input("Enter CSV file to calibrate: ").strip()

if not os.path.exists(filename):
    raise FileNotFoundError(f"File '{filename}' not found.")

# ===========================
# Load CSV, skip first 2 header lines
# ===========================
try:
    df = pd.read_csv(filename, skiprows=2)
except pd.errors.EmptyDataError:
    raise ValueError(f"No data found in {filename}. Make sure you have collected samples.")

if df.empty:
    raise ValueError(f"No data rows found in {filename}. Collect some samples before calibrating.")

print(f"Columns detected: {df.columns.tolist()}")
print(f"Number of data rows: {len(df)}")

# ===========================
# Separate FT (target) and capacitance (features)
# ===========================
target_cols = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
feature_cols = ["C1", "C2", "C3", "C4"]

for col in target_cols + feature_cols:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found in CSV.")

X = df[feature_cols].values  # capacitance readings
y = df[target_cols].values   # measured FT values

# ===========================
# Fit linear regression
# ===========================
model = LinearRegression()
model.fit(X, y)

# Predict to check fit
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"\nLinear regression fitted. RMSE: {rmse:.4f}")

# ===========================
# Show coefficients
# ===========================
print("\nCalibration coefficients (linear model):")
for i, col in enumerate(feature_cols):
    coeffs = model.coef_[:, i]
    print(f"{col}: Fx={coeffs[0]:.4f}, Fy={coeffs[1]:.4f}, Fz={coeffs[2]:.4f}, "
          f"Tx={coeffs[3]:.4f}, Ty={coeffs[4]:.4f}, Tz={coeffs[5]:.4f}")

print("\nIntercepts:")
print(f"Fx, Fy, Fz, Tx, Ty, Tz = {model.intercept_}")

# ===========================
# Save model
# ===========================
model_filename = filename.replace(".csv", "_linear_model.pkl")
joblib.dump(model, model_filename)
print(f"\nLinear model saved to '{model_filename}'")
