import pandas as pd
from datetime import datetime

# ------------------------------
# Airport Mapping & Holiday Functions
# ------------------------------

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
    # Other mappings...
    "LHR": "United Kingdom", "LGW": "United Kingdom",
    "CDG": "France",
    "AMS": "Netherlands",
    "FRA": "Germany", "MUC": "Germany",
    "DUB": "Ireland",
    "MAD": "Spain", "BCN": "Spain",
    "FCO": "Italy",
    "ZRH": "Switzerland",
    "LIS": "Portugal",
    "BRU": "Belgium",
    "ARN": "Sweden",
    "OSL": "Norway",
    "CPH": "Denmark",
    "VIE": "Austria",
    "PRG": "Czech Republic",
    "ATH": "Greece",
    "HEL": "Finland",
    "HND": "Japan", "NRT": "Japan", "KIX": "Japan",
    "ICN": "South Korea",
    "PVG": "China", "PEK": "China",
    "HKG": "Hong Kong",
    "SIN": "Singapore",
    "BKK": "Thailand",
    "KUL": "Malaysia",
    "MNL": "Philippines",
    "DEL": "India", "BOM": "India",
    "DXB": "United Arab Emirates",
    "DOH": "Qatar",
    "MCT": "Oman",
    "CUN": "Mexico", "MEX": "Mexico",
    "PUJ": "Dominican Republic",
    "MBJ": "Jamaica",
    "HAV": "Cuba",
    "LIM": "Peru",
    "GRU": "Brazil",
    "BOG": "Colombia",
    "SCL": "Chile",
    "SYD": "Australia", "MEL": "Australia",
    "AKL": "New Zealand",
    "CPT": "South Africa", "JNB": "South Africa",
    "CMN": "Morocco",
    "ALG": "Algeria",
    "CAI": "Egypt"
}
holiday_csv_path = r"C:\Users\Meyssa\OneDrive - University of Ottawa\Desktop\MIA5100 W00 Fndn & App. Machine Learning\Flight_price_prediction\Best booking date\Holidays.csv"

def get_country(airport_code):
    return airport_to_country.get(airport_code.upper(), "Unknown")

def parse_holiday_date(holiday_date_str):
    """
    Parse a holiday date string that may be in "MM/DD/YYYY" or "MM/DD" format.
    For date ranges separated by '-', both dates are parsed.
    """
    if "-" in holiday_date_str:
        start_str, end_str = holiday_date_str.split("-")
        start_str = start_str.strip()
        end_str = end_str.strip()
        try:
            start_dt = datetime.strptime(start_str, "%m/%d/%Y")
        except ValueError:
            start_dt = datetime.strptime(start_str, "%m/%d").replace(year=datetime.today().year)
        try:
            end_dt = datetime.strptime(end_str, "%m/%d/%Y")
        except ValueError:
            end_dt = datetime.strptime(end_str, "%m/%d").replace(year=datetime.today().year)
    else:
        s = holiday_date_str.strip()
        try:
            start_dt = datetime.strptime(s, "%m/%d/%Y")
        except ValueError:
            start_dt = datetime.strptime(s, "%m/%d").replace(year=datetime.today().year)
        end_dt = start_dt
    return start_dt, end_dt

def load_holiday_data(holiday_csv_path):
    """
    Load the holiday CSV file and return a list of holiday records.
    Each record is a dictionary with the country, start and end dates, and holiday/peak flags.
    """
    df_holidays = pd.read_csv(holiday_csv_path)
    records = []
    for _, row in df_holidays.iterrows():
        country = str(row["Country"]).strip()
        holiday_date_str = str(row["Holiday Date"]).strip()
        hol_season = str(row["holiday season?"]).strip()
        peak_season = str(row["Peak Season?"]).strip()
        start_dt, end_dt = parse_holiday_date(holiday_date_str)
        records.append({
            "country": country,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "holiday_season": hol_season,
            "peak_season": peak_season
        })
    return records

def check_holiday_peak(country, flight_date, holiday_records):
    """
    For a given country and flight date (in "YYYY-MM-DD" format), determine if the date falls
    within a holiday or peak period based on the holiday records.
    Returns a tuple (holiday_flag, peak_flag) where each is "Yes" or "No".
    """
    try:
        flight_date_dt = datetime.strptime(flight_date, "%Y-%m-%d")
    except Exception:
        return "No", "No"
    
    holiday_flag = "No"
    peak_flag = "No"
    for record in holiday_records:
        # Check if the record matches the country (case-insensitive)
        if record["country"].lower() == country.lower():
            if record["start_dt"] <= flight_date_dt <= record["end_dt"]:
                if record["holiday_season"].strip().lower() == "yes":
                    holiday_flag = "Yes"
                if record["peak_season"].strip().lower() == "yes":
                    peak_flag = "Yes"
    return holiday_flag, peak_flag

# ------------------------------
# Process Flights Data
# ------------------------------

# Load your flights CSV (adjust the file path as needed)
flights_df = pd.read_csv("combined_route_info.csv")

# Convert date columns to datetime objects
flights_df["Booking Date"] = pd.to_datetime(flights_df["Booking Date"], errors="coerce")
flights_df["Departure Date"] = pd.to_datetime(flights_df["Departure Date"], errors="coerce")

# Calculate Booking Lead Time if missing (difference in days between Departure Date and Booking Date)
mask_missing_lead = flights_df["Booking Lead Time"].isnull() | (flights_df["Booking Lead Time"] == "")
flights_df.loc[mask_missing_lead, "Booking Lead Time"] = (
    flights_df.loc[mask_missing_lead, "Departure Date"] - flights_df.loc[mask_missing_lead, "Booking Date"]
).dt.days

# Load holiday mapping records from holidays.csv (adjust the file path as needed)
holiday_records = load_holiday_data(holiday_csv_path)

# Function to update holiday/peak season columns for a given row
def update_holiday_flags(row):
    # Process Origin holiday/peak flags if missing
    if pd.isnull(row["Origin_Holiday_Season"]) or pd.isnull(row["Origin_Peak_Season"]):
        origin_country = get_country(row["Origin"])
        if pd.notnull(row["Departure Date"]):
            flight_date_str = row["Departure Date"].strftime("%Y-%m-%d")
            origin_hol, origin_peak = check_holiday_peak(origin_country, flight_date_str, holiday_records)
        else:
            origin_hol, origin_peak = "No", "No"
        row["Origin_Holiday_Season"] = origin_hol
        row["Origin_Peak_Season"] = origin_peak
    else:
        origin_hol = row["Origin_Holiday_Season"]
        origin_peak = row["Origin_Peak_Season"]
    
    # Process Destination holiday/peak flags if missing
    if pd.isnull(row["Dest_Holiday_Season"]) or pd.isnull(row["Dest_Peak_Season"]):
        dest_country = get_country(row["Destination"])
        if pd.notnull(row["Departure Date"]):
            flight_date_str = row["Departure Date"].strftime("%Y-%m-%d")
            dest_hol, dest_peak = check_holiday_peak(dest_country, flight_date_str, holiday_records)
        else:
            dest_hol, dest_peak = "No", "No"
        row["Dest_Holiday_Season"] = dest_hol
        row["Dest_Peak_Season"] = dest_peak
    else:
        dest_hol = row["Dest_Holiday_Season"]
        dest_peak = row["Dest_Peak_Season"]
    
    # For On_Holiday and On_Peak, mark "Yes" if either origin or destination is in holiday/peak
    row["On_Holiday"] = "Yes" if (origin_hol == "Yes" or dest_hol == "Yes") else "No"
    row["On_Peak"] = "Yes" if (origin_peak == "Yes" or dest_peak == "Yes") else "No"
    return row

# Apply the update function to each row in the DataFrame
flights_df = flights_df.apply(update_holiday_flags, axis=1)

# Save the updated DataFrame to a new CSV file
flights_df.to_csv("dataset.csv", index=False)

print("CSV processing complete. The updated file is saved as 'dataset.csv'.")
