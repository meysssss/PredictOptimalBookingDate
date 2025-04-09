import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import category_encoders as ce  # For target encoding
import joblib

# --------------------------------------
# Airport to Country Mapping Dictionary
# --------------------------------------
airport_to_country = {
    # Canada
    "YYZ": "Canada", "YUL": "Canada", "YVR": "Canada", "YYC": "Canada", "YEG": "Canada",
    "YOW": "Canada", "YWG": "Canada", "YHZ": "Canada", "YQB": "Canada", "YQT": "Canada",
    "YXE": "Canada", "YQG": "Canada", "YYG": "Canada", "YHM": "Canada", "YQR": "Canada",
    "YFB": "Canada", "YBB": "Canada", "YUB": "Canada", "YFC": "Canada", "YMM": "Canada",
    "YZF": "Canada", "YQX": "Canada", "YAM": "Canada", "YDF": "Canada", "YXU": "Canada",
    "YKF": "Canada", "YQZ": "Canada", "YPG": "Canada", "YRD": "Canada", "YSC": "Canada",
    "YQL": "Canada", "YXJ": "Canada",
    # USA
    "JFK": "USA", "LGA": "USA", "EWR": "USA", "BOS": "USA",
    "ORD": "USA", "DTW": "USA", "ATL": "USA", "MIA": "USA",
    "DFW": "USA", "LAX": "USA", "SFO": "USA", "DEN": "USA",
    "IAD": "USA", "DCA": "USA", "PHX": "USA", "MSP": "USA",
    "CLE": "USA", "MCO": "USA", "IAH": "USA", "STL": "USA",
    "PIT": "USA", "SLC": "USA", "LAS": "USA", "SEA": "USA",
    # Puerto Rico
    "SJU": "Puerto Rico",
    # United Kingdom
    "LHR": "United Kingdom", "LGW": "United Kingdom",
    # France
    "CDG": "France",
    # Netherlands
    "AMS": "Netherlands",
    # Germany
    "FRA": "Germany", "MUC": "Germany",
    # Ireland
    "DUB": "Ireland",
    # Spain
    "MAD": "Spain", "BCN": "Spain",
    # Italy
    "FCO": "Italy",
    # Switzerland
    "ZRH": "Switzerland",
    # Portugal
    "LIS": "Portugal",
    # Belgium
    "BRU": "Belgium",
    # Sweden
    "ARN": "Sweden",
    # Norway
    "OSL": "Norway",
    # Denmark
    "CPH": "Denmark",
    # Austria
    "VIE": "Austria",
    # Czech Republic
    "PRG": "Czech Republic",
    # Greece
    "ATH": "Greece",
    # Finland
    "HEL": "Finland",
    # Japan
    "HND": "Japan", "NRT": "Japan", "KIX": "Japan",
    # South Korea
    "ICN": "South Korea",
    # China
    "PVG": "China", "PEK": "China",
    # Hong Kong
    "HKG": "Hong Kong",
    # Singapore
    "SIN": "Singapore",
    # Thailand
    "BKK": "Thailand",
    # Malaysia
    "KUL": "Malaysia",
    # Philippines
    "MNL": "Philippines",
    # India
    "DEL": "India", "BOM": "India",
    # United Arab Emirates
    "DXB": "United Arab Emirates",
    # Qatar
    "DOH": "Qatar",
    # Oman
    "MCT": "Oman",
    # Mexico
    "CUN": "Mexico", "MEX": "Mexico",
    # Dominican Republic
    "PUJ": "Dominican Republic",
    # Jamaica
    "MBJ": "Jamaica",
    # Cuba
    "HAV": "Cuba",
    # Peru
    "LIM": "Peru",
    # Brazil
    "GRU": "Brazil",
    # Colombia
    "BOG": "Colombia",
    # Chile
    "SCL": "Chile",
    # Australia
    "SYD": "Australia", "MEL": "Australia",
    # New Zealand
    "AKL": "New Zealand",
    # South Africa
    "CPT": "South Africa", "JNB": "South Africa",
    # Morocco
    "CMN": "Morocco",
    # Algeria
    "ALG": "Algeria",
    # Egypt
    "CAI": "Egypt"
}

# --------------------------------------
# Helper Functions for Holiday Data Processing
# --------------------------------------
def parse_date_range(date_str):
    date_str = date_str.strip()
    if '-' in date_str:
        parts = date_str.split('-')
        if len(parts[0].split('/')) == 2:
            year = parts[1].split('/')[-1]
            start_date_str = parts[0].strip() + '/' + year
            end_date_str = parts[1].strip()
        else:
            start_date_str = parts[0].strip()
            end_date_str = parts[1].strip()
    else:
        start_date_str = date_str
        end_date_str = date_str
    start_date = pd.to_datetime(start_date_str, errors='coerce')
    end_date = pd.to_datetime(end_date_str, errors='coerce')
    return start_date, end_date

def load_and_process_holidays(holiday_file):
    holiday_df = pd.read_csv(holiday_file)
    holiday_df = holiday_df.loc[:, ~holiday_df.columns.str.contains('^Unnamed')]
    holiday_df[['Start_Date', 'End_Date']] = holiday_df['Holiday Date'].apply(
        lambda x: pd.Series(parse_date_range(x))
    )
    return holiday_df

def fill_holiday_flags(df, holiday_df, airport_to_country):
    if 'On_Holiday' not in df.columns:
        df['On_Holiday'] = np.nan
    if 'On_Peak' not in df.columns:
        df['On_Peak'] = np.nan
    for i, row in df.iterrows():
        if pd.notna(row['On_Holiday']) and pd.notna(row['On_Peak']):
            continue
        country = airport_to_country.get(row['Origin'], None)
        if not country:
            country = airport_to_country.get(row['Destination'], None)
        if not country:
            df.at[i, 'On_Holiday'] = 0
            df.at[i, 'On_Peak'] = 0
            continue
        flight_date = row['Departure Date']
        on_holiday = 0
        on_peak = 0
        holidays_country = holiday_df[holiday_df['Country'] == country]
        for j, holiday_row in holidays_country.iterrows():
            start_date = holiday_row['Start_Date']
            end_date = holiday_row['End_Date']
            if pd.isna(start_date) or pd.isna(end_date):
                continue
            if start_date <= flight_date <= end_date:
                if str(holiday_row['Holiday season?']).strip().lower() == 'yes':
                    on_holiday = 1
                if str(holiday_row['Peak Season?']).strip().lower() == 'yes':
                    on_peak = 1
        df.at[i, 'On_Holiday'] = on_holiday
        df.at[i, 'On_Peak'] = on_peak
    df['On_Holiday'] = df['On_Holiday'].fillna(0).astype(int)
    df['On_Peak'] = df['On_Peak'].fillna(0).astype(int)
    return df

# --------------------------------------
# Flight Data Processing Functions
# --------------------------------------
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    critical_columns = ['Price', 'Booking Date', 'Departure Date', 'Origin', 'Destination']
    df.dropna(subset=critical_columns, inplace=True)
    df.drop_duplicates(inplace=True)
    irrelevant_cols = ["Origin_Holiday_Season", "Origin_Peak_Season",
                       "Dest_Holiday_Season", "Dest_Peak_Season"]
    df.drop(columns=irrelevant_cols, inplace=True, errors='ignore')
    return df

def convert_dates(df):
    for col in ['Booking Date', 'Departure Date', 'Arrival Date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['Booking_Month'] = df['Booking Date'].dt.month
    df['Booking_DayOfWeek'] = df['Booking Date'].dt.dayofweek
    df['Departure_Month'] = df['Departure Date'].dt.month
    df['Arrival_Month'] = df['Arrival Date'].dt.month
    df['Days_Between'] = (df['Departure Date'] - df['Booking Date']).dt.days
    return df

def extract_stops(text):
    text = str(text).lower()
    if "direct" in text or "nonstop" in text:
        return 0
    match = re.search(r'(\d+)', text)
    if match:
        return int(match.group(1))
    return np.nan

def process_stops(df):
    df['Num_Stops_Outbound'] = df['stops - outbound'].apply(extract_stops)
    df['Num_Stops_Return'] = df['stops - return'].apply(extract_stops)
    df['Total_Stops'] = df['Num_Stops_Outbound'] + df['Num_Stops_Return']
    return df

def filter_max_stops(df, max_stops=3):
    return df[df['Total_Stops'] <= max_stops]

def classify_route_type(df):
    canadian_airports = {
        "YYZ", "YUL", "YVR", "YYC", "YEG", "YOW", "YWG", "YHZ", "YQB", "YQT",
        "YXE", "YQG", "YYG", "YHM", "YQR", "YFB", "YBB", "YUB", "YFC", "YMM",
        "YZF", "YQX", "YAM", "YDF", "YXU", "YKF", "YQZ", "YPG", "YRD", "YSC",
        "YQL", "YXJ"
    }
    df['Route_Type'] = df.apply(
        lambda row: "Domestic" if row['Origin'] in canadian_airports and row['Destination'] in canadian_airports 
        else "International", axis=1)
    return df

def process_flags(df):
    for col in ['On_Holiday', 'On_Peak']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

def clean_airline(airline):
    if pd.isnull(airline):
        return airline
    if '•' in airline:
        airline = airline.split('•')[0]
    if ',' in airline:
        airline = airline.split(',')[0]
    return airline.strip()

def process_airline(df):
    df['Airline_clean'] = df['Airline'].apply(clean_airline)
    mask = df['Airline_clean'].str.contains('bus', case=False) | df['Airline_clean'].str.contains('via rail', case=False)
    df = df[~mask]
    return df

def duration_to_minutes(duration_str):
    if pd.isnull(duration_str):
        return np.nan
    hours = 0
    minutes = 0
    hour_match = re.search(r'(\d+)\s*h', duration_str)
    minute_match = re.search(r'(\d+)\s*m', duration_str)
    if hour_match:
        hours = int(hour_match.group(1))
    if minute_match:
        minutes = int(minute_match.group(1))
    return hours * 60 + minutes

def process_duration(df):
    df['duration_inbound_minutes'] = df['duration - inbound'].apply(duration_to_minutes)
    df['duration_return_minutes'] = df['duration - return'].apply(duration_to_minutes)
    return df

def drop_unnecessary_columns(df):
    columns_to_drop = ['Departure Day', 'Arrival Day', 'duration - inbound', 'duration - return']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return df

# --------------------------------------
# New Function: Add "Route" Feature and Apply Target Encoding
# --------------------------------------
def add_and_target_encode_route(df):
    # Create the Route feature by concatenating Origin and Destination
    df['Route'] = df['Origin'].str.strip() + '-' + df['Destination'].str.strip()
    # Initialize the TargetEncoder for the Route column
    encoder = ce.TargetEncoder(cols=['Route'])
    # Fit and transform the Route feature using Price as the target
    df['Route_encoded'] = encoder.fit_transform(df['Route'], df['Price'])
    # Save the encoder object for future use (e.g., in predictions)
    joblib.dump(encoder, 'route_target_encoder.pkl')
    print("✅ Target encoder saved as 'route_target_encoder.pkl'")
    return df

# --------------------------------------
# EDA and Categorical Encoding Functions
# --------------------------------------
def run_eda(df):
    num_features = ['Price', 'Trip Duration', 'Days_Between',
                    'Booking_Month', 'Booking_DayOfWeek',
                    'Departure_Month', 'Arrival_Month',
                    'Total_Stops', 'On_Holiday', 'On_Peak']
    plt.figure(figsize=(12, 10))
    corr_matrix = df[num_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix for Key Features")
    plt.tight_layout()
    plt.show()
    
    agg_origin = df.groupby(['Origin', 'Departure_Month'])['Price'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=agg_origin, x='Departure_Month', y='Price', hue='Origin', marker='o')
    plt.title("Average Ticket Price by Departure Month per Origin")
    plt.xlabel("Departure Month")
    plt.ylabel("Average Price")
    plt.legend(title="Origin", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    agg_booking_day = df.groupby('Booking_DayOfWeek')['Price'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg_booking_day, x='Booking_DayOfWeek', y='Price', palette="viridis")
    plt.title("Average Ticket Price by Booking Day of Week")
    plt.xlabel("Booking Day of Week (0=Mon, 6=Sun)")
    plt.ylabel("Average Price")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Total_Stops', y='Price', data=df, palette="Set3")
    plt.title("Ticket Price Distribution by Total Number of Stops")
    plt.xlabel("Total Stops (Outbound + Return)")
    plt.ylabel("Ticket Price")
    plt.tight_layout()
    plt.show()
    
    agg_days = df.groupby('Days_Between')['Price'].agg(['mean', 'median']).reset_index()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=agg_days, x='Days_Between', y='mean', label='Mean Price')
    sns.lineplot(data=agg_days, x='Days_Between', y='median', color='red', label='Median Price')
    plt.title("Ticket Price vs. Days Between Booking and Departure")
    plt.xlabel("Days Between Booking and Departure")
    plt.ylabel("Ticket Price")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='On_Peak', y='Price', data=df, palette="Pastel1")
    plt.title("Ticket Price Distribution: On Peak vs. Not Peak")
    plt.xlabel("On Peak (1=Yes, 0=No)")
    plt.ylabel("Ticket Price")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='On_Holiday', y='Price', data=df, palette="Pastel2")
    plt.title("Ticket Price Distribution: On Holiday vs. Not Holiday")
    plt.xlabel("On Holiday (1=Yes, 0=No)")
    plt.ylabel("Ticket Price")
    plt.tight_layout()
    plt.show()
    
    agg_route = df.groupby('Route_Type')['Price'].agg(['mean', 'median']).reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(data=agg_route, x='Route_Type', y='mean', palette="coolwarm")
    plt.title("Average Ticket Price by Route Type")
    plt.xlabel("Route Type")
    plt.ylabel("Average Ticket Price")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Route_Type', y='Price', palette="coolwarm")
    plt.title("Ticket Price Distribution by Route Type")
    plt.xlabel("Route Type")
    plt.ylabel("Ticket Price")
    plt.tight_layout()
    plt.show()

def encode_categorical(df):
    origin_dummies = pd.get_dummies(df['Origin'], prefix='Origin')
    destination_dummies = pd.get_dummies(df['Destination'], prefix='Destination')
    airline_dummies = pd.get_dummies(df['Airline_clean'], prefix='Airline')
    df = pd.concat([df, origin_dummies, destination_dummies, airline_dummies], axis=1)
    df.drop(columns=['Origin', 'Destination', 'Airline', 'Airline_clean'], inplace=True)
    return df

def label_encode_stops(df):
    le_outbound = LabelEncoder()
    le_return = LabelEncoder()
    df['stops_outbound_encoded'] = le_outbound.fit_transform(df['stops - outbound'])
    df['stops_return_encoded'] = le_return.fit_transform(df['stops - return'])
    df.drop(columns=['stops - outbound', 'stops - return'], inplace=True)
    return df

# --------------------------------------
# Pipeline Functions
# --------------------------------------
def process_basic(file_path):
    df = load_and_clean_data(file_path)
    df = convert_dates(df)
    df = process_stops(df)
    df = filter_max_stops(df, max_stops=3)
    df = classify_route_type(df)
    df = process_flags(df)
    df = process_airline(df)
    df = process_duration(df)
    return df

def prepare_for_model(df):
    df = drop_unnecessary_columns(df)
    # Add Route feature and apply target encoding
    df = add_and_target_encode_route(df)
    df = encode_categorical(df)
    df = label_encode_stops(df)
    df['Route_Type'] = df['Route_Type'].map({'Domestic': 0, 'International': 1})
    return df

# --------------------------------------
# Main Execution
# --------------------------------------
if __name__ == "__main__":
    flight_data_file = "combined_dataset_.csv"  # Update with your flight data CSV
    holiday_data_file = "holidays.csv"           # Update with your holiday CSV
    
    # Process flight data
    df_basic = process_basic(flight_data_file)
    
    # Load and process holiday data
    holiday_df = load_and_process_holidays(holiday_data_file)
    
    # Fill missing On_Holiday and On_Peak flags using the holiday mapping
    df_basic = fill_holiday_flags(df_basic, holiday_df, airport_to_country)
    
    # Run EDA to visualize data
    run_eda(df_basic)
    
    # Prepare data for modeling (including Route target encoding)
    df_model = prepare_for_model(df_basic.copy())
    
    # Split into training and test sets (80/20 split)
    train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=42)
    
    # Save the training and test sets as CSV files
    train_csv_file = "cleaned_flight_booking_data_train.csv"
    test_csv_file = "cleaned_flight_booking_data_test.csv"
    train_df.to_csv(train_csv_file, index=False)
    test_df.to_csv(test_csv_file, index=False)
    
    print("Processed training data saved to:", train_csv_file)
    print("Processed test data saved to:", test_csv_file)
