import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("restaurant_customer_satisfaction (1).csv")
    return data

data = load_data()

# -------------------------------
# Preprocessing
# -------------------------------
data.fillna(method="ffill", inplace=True)

# Drop ID column if present
if "CustomerID" in data.columns:
    data.drop("CustomerID", axis=1, inplace=True)

# Label Encoding
categorical_cols = [
    "Gender",
    "VisitFrequency",
    "PreferredCuisine",
    "TimeOfVisit",
    "DiningOccasion",
    "MealType"
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features & Target
X = data.drop("HighSatisfaction", axis=1)
y = data["HighSatisfaction"]

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

feature_columns = X.columns

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🍽️ Restaurant Customer Satisfaction Prediction")
st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 70, 25)
gender = st.sidebar.selectbox("Gender", label_encoders["Gender"].classes_)
income = st.sidebar.number_input("Income", 10000, 200000, 50000)

visit = st.sidebar.selectbox(
    "Visit Frequency", label_encoders["VisitFrequency"].classes_
)
cuisine = st.sidebar.selectbox(
    "Preferred Cuisine", label_encoders["PreferredCuisine"].classes_
)
time = st.sidebar.selectbox(
    "Time of Visit", label_encoders["TimeOfVisit"].classes_
)
occasion = st.sidebar.selectbox(
    "Dining Occasion", label_encoders["DiningOccasion"].classes_
)

group_size = st.sidebar.slider("Group Size", 1, 10, 2)
meal = st.sidebar.selectbox(
    "Meal Type", label_encoders["MealType"].classes_
)

loyalty = st.sidebar.selectbox("Loyalty Program Member", ["Yes", "No"])
avg_spend = st.sidebar.number_input("Average Spend", 100, 10000, 1000)
service = st.sidebar.slider("Service Rating (1–5)", 1, 5, 3)
food = st.sidebar.slider("Food Rating (1–5)", 1, 5, 3)
ambiance = st.sidebar.slider("Ambiance Rating (1–5)", 1, 5, 3)
delivery = st.sidebar.selectbox("Delivery Order", ["Yes", "No"])
online_reservation = st.sidebar.selectbox("Online Reservation", ["Yes", "No"])
wait_time = st.sidebar.slider("Wait Time (minutes)", 0, 60, 10)

# Binary conversions
loyalty = 1 if loyalty == "Yes" else 0
delivery = 1 if delivery == "Yes" else 0
online_reservation = 1 if online_reservation == "Yes" else 0

# -------------------------------
# Input DataFrame
# -------------------------------
input_dict = {
    "Age": age,
    "Gender": label_encoders["Gender"].transform([gender])[0],
    "Income": income,
    "VisitFrequency": label_encoders["VisitFrequency"].transform([visit])[0],
    "PreferredCuisine": label_encoders["PreferredCuisine"].transform([cuisine])[0],
    "TimeOfVisit": label_encoders["TimeOfVisit"].transform([time])[0],
    "DiningOccasion": label_encoders["DiningOccasion"].transform([occasion])[0],
    "GroupSize": group_size,
    "MealType": label_encoders["MealType"].transform([meal])[0],
    "LoyaltyProgramMember": loyalty,
    "AverageSpend": avg_spend,
    "ServiceRating": service,
    "FoodRating": food,
    "AmbianceRating": ambiance,
    "DeliveryOrder": delivery,
    "OnlineReservation": online_reservation,
    "WaitTime": wait_time
}

input_data = pd.DataFrame([input_dict])

# Ensure correct column order
input_data = input_data[feature_columns]

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Customer is Highly Satisfied")
    else:
        st.error("❌ Customer is Not Highly Satisfied")
