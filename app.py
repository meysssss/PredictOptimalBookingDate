from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

app = Flask(__name__)
CORS(app)

model = joblib.load("stacking_ensemble_model.pkl")

def build_features(origin, destination, dep_date, arr_date, lead_days):
    features = {
        'Trip Duration': (arr_date - dep_date).days,
        'On_Holiday': 0,
        'On_Peak': 0,
        'Booking_Month': dep_date.month,
        'Booking_DayOfWeek': (dep_date - timedelta(days=lead_days)).weekday(),
        'Departure_Month': dep_date.month,
        'Arrival_Month': arr_date.month,
        'Days_Between': lead_days,
        'Num_Stops_Outbound': 1,
        'Num_Stops_Return': 1,
        'Total_Stops': 2,
        'Route_Type': 0 if origin.startswith('Y') and destination.startswith('Y') else 1,
        'duration_inbound_minutes': 300,
        'duration_return_minutes': 300,
        'Route_encoded': hash(f"{origin}_{destination}") % 10000,
    }

    for code in ['YEG', 'YUL', 'YVR', 'YYC', 'YYZ']:
        features[f"Origin_{code}"] = 1 if origin == code else 0

    all_dests = [
        'YYZ', 'YVR', 'YUL', 'YYC', 'YEG',
        'BCN', 'FCO', 'ZRH', 'LIS', 'BRU', 'ARN', 'OSL', 'CPH', 'VIE', 'PRG',
        'ATH', 'HND', 'NRT', 'ICN', 'HKG', 'SIN', 'BKK', 'DEL', 'BOM', 'DXB',
        'DOH', 'JFK', 'ORD', 'LAX', 'SFO'
    ]
    for code in all_dests:
        features[f"Destination_{code}"] = 1 if destination == code else 0

    input_df = pd.DataFrame([features])
    input_df.columns = input_df.columns.map(str)
    input_df = input_df.reindex(columns=[str(col) for col in model.feature_names_in_], fill_value=0)
    return input_df

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    origin = data["origin"].upper()
    destination = data["destination"].upper()
    dep_date = pd.to_datetime(data["departure_date"])
    arr_date = pd.to_datetime(data["arrival_date"])

    results_by_day = []
    for lead in range(10, 180):
        X = build_features(origin, destination, dep_date, arr_date, lead)
        price = model.predict(X)[0]
        results_by_day.append((lead, round(price, 2)))

    results_by_duration = []
    for duration in range(3, 31):
        new_arr = dep_date + timedelta(days=duration)
        X = build_features(origin, destination, dep_date, new_arr, 30)
        price = model.predict(X)[0]
        results_by_duration.append((duration, round(price, 2)))

    calendar_prices = []
    for d_shift in range(-3, 4):
        for r_shift in range(-3, 4):
            shifted_dep = dep_date + timedelta(days=d_shift)
            shifted_arr = arr_date + timedelta(days=r_shift)
            if shifted_arr > shifted_dep:
                X = build_features(origin, destination, shifted_dep, shifted_arr, 30)
                price = model.predict(X)[0]
                calendar_prices.append({
                    "dep": shifted_dep.strftime('%Y-%m-%d'),
                    "ret": shifted_arr.strftime('%Y-%m-%d'),
                    "price": round(price, 2)
                })

    best_day = min(results_by_day, key=lambda x: x[1])
    best_booking_date = (dep_date - timedelta(days=best_day[0])).strftime('%Y-%m-%d')
    predicted_price = round(best_day[1], 2)

    return jsonify({
        "best_booking_date": best_booking_date,
        "predicted_price": predicted_price,
        "price_by_day": results_by_day,
        "price_by_duration": results_by_duration,
        "calendar_prices": calendar_prices
    })

if __name__ == "__main__":
    app.run(debug=True)
