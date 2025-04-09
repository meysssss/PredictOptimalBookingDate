import pandas as pd 
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# --- Configuration ---
model_path = "stacking_ensemble_model.pkl"
test_data_path = r"C:\Users\Meyssa\OneDrive - University of Ottawa\Desktop\MIA5100 W00 Fndn & App. Machine Learning\Flight_price_prediction\Flight_price_prediction\Best booking date\cleaned_flight_booking_data_test.csv"

# --- Load Model ---
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at: {model_path}")
print("✅ Loaded stacking ensemble model")
model = joblib.load(model_path)

# --- Load Test Data ---
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"❌ Test CSV not found at: {test_data_path}")
print("📥 Loading test data...")

df_test = pd.read_csv(test_data_path)
y_test = df_test["Price"]

# Drop non-feature columns if they exist
drop_cols = ["Price", "Booking Date", "Departure Date", "Arrival Date", "Route"]
X_test = df_test.drop(columns=drop_cols, errors="ignore")

# --- Predict ---
print("🤖 Running predictions on test set...")
y_pred = model.predict(X_test)

# --- Evaluate ---
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --- Report ---
print("\n📊 Test Set Evaluation for Stacking Ensemble Model")
print("-----------------------------------------------")
print(f"✅ MAE  (Mean Absolute Error):     {mae:.2f}")
print(f"✅ RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"✅ R²   (R-squared Score):         {r2:.3f}")
print("-----------------------------------------------")
