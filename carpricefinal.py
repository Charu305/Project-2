import streamlit as st
import pickle
import joblib
import pandas as pd

# --- Load Assets ---
model = joblib.load("C:/Users/babuk/Downloads/lightgbm_price_model.pkl")

with open("C:/Users/babuk/Downloads/label_encoded.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("C:/Users/babuk/Downloads/one_hot_encoded.pkl", "rb") as f:
    one_hot_structure = pickle.load(f)

with open("C:/Users/babuk/Downloads/target_encoding_map.pkl", "rb") as f:
    target_encoding_map = pickle.load(f)

# --- Load full dataset to enable dependent dropdowns ---
#df = pd.read_csv("C:/Users/babuk/Downloads/car.csv")
df = pd.read_excel("C:/Users/babuk/OneDrive/Desktop/car.xlsx")

# --- Target Encoding Function ---
def apply_target_encoding(df, encoding_map):
    for col, mapping in encoding_map.items():
        df[col + '_target'] = df[col].map(mapping)
    return df

# --- UI ---
st.title("🚗 Car Price Prediction")
st.header("Enter Car Details")

# --- Step 1: Select OEM ---
oem = st.selectbox("OEM", sorted(df['oem'].dropna().unique()))

# --- Step 2: Filter dataset based on selected OEM ---
filtered_df = df[df['oem'] == oem]

# --- Step 3: Dynamically generate dropdown options ---
body_types = sorted(filtered_df['bt'].dropna().unique())
fuel_types = sorted(filtered_df['ft'].dropna().unique())
cities = sorted(filtered_df['City'].dropna().unique())
transmissions = sorted(filtered_df['transmission'].dropna().unique())

# --- Step 4: Other inputs ---
bt = st.selectbox("Body Type", body_types)
model_year = st.slider("Model Year", min_value=2000, max_value=2025, value=2019)
engine = st.number_input("Engine (cc)", min_value=600, max_value=6000, step=100)
mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=50.0, step=0.5)
kms_driven = st.number_input("Kms Driven", min_value=0, step=1000)
transmission = st.selectbox("Transmission", transmissions)
fuel_type = st.selectbox("Fuel Type", fuel_types)
city = st.selectbox("City", cities)

# --- Step 5: Build Input Data ---
input_data = {
    "modelYear": model_year,
    "engine": engine,
    "mileage": mileage,
    "kms_driven": kms_driven,
    "transmission_encoded": 1 if transmission == "Manual" else 0,
}

# --- Step 6: One-hot encode ---
for col in one_hot_structure.columns:
    input_data[col] = 0
input_data[f"ft_{fuel_type}"] = 1
input_data[f"City_{city}"] = 1
input_data[f"bt_{bt}"] = 1

# --- Step 7: Target encoding ---
input_df = pd.DataFrame([input_data])
input_df["oem"] = oem  # Add for target encoding
input_df = apply_target_encoding(input_df, target_encoding_map)
input_df.drop(columns=["oem"], inplace=True)

# --- Step 8: Align with model input features ---
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# --- Step 9: Predict ---
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]  # Price in rupees

    if prediction >= 1_00_00_000:
        formatted_price = f"₹{prediction / 1_00_00_000:.2f} Cr"
    elif 10_00_000 <= prediction < 1_00_00_000:
        formatted_price = f"₹{prediction / 1_00_000:.2f} Lakh"
    elif 1_00_000 <= prediction < 10_00_000:
        formatted_price = f"₹{prediction / 1_00_000:.2f} Lakh"
    elif 1_000 <= prediction < 1_00_000:
        formatted_price = f"₹{prediction / 1_000:.2f} Thousand"
    else:
        formatted_price = f"₹{prediction:.2f}"

    st.success(f"Estimated Car Price: {formatted_price}")
