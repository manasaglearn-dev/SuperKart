import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ===============================
# Load Model from Hugging Face
# ===============================
model_path = hf_hub_download(
    repo_id="armakar123/superkart-sales-model",
    filename="best_superkart_model_v1.joblib"
)

model = joblib.load(model_path)

# ===============================
# Streamlit UI
# ===============================
st.title("SuperKart Sales Prediction App")

st.write("""
This application predicts the **total sales of a product in a store**
based on product and store-related attributes.
""")

# ===============================
# User Inputs
# ===============================
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=1.0)
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=1.0)
Product_MRP = st.number_input("Product MRP", min_value=1.0, value=10.0)
Store_Establishment_Year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, value=2005)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

Product_Type = st.selectbox(
    "Product Type",
    ["Dairy", "Health and Hygiene", "Meat", "Breads", "Fruits and Vegetables", "Household","Baking Goods","Breakfast","Canned","Frozen Foods","Hard Drinks","Others","Seafoods","Soft Drinks","Snack Foods","Starchy Foods"]
)

Store_Id = st.text_input("Store ID", ["OUT001","OUT002","OUT003","OUT004"])
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox(
    "Store Type",
    ["Supermarket Type1", "Supermarket Type2", "Food Mart","Departmental Store"]
)

# ===============================
# Assemble Input Data
# ===============================
input_data = pd.DataFrame([{
    "Product_Weight": Product_Weight,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Type": Product_Type,
    "Store_Id": Store_Id,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type
}])

# ===============================
# Prediction
# ===============================
if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]

    st.subheader("📊 Predicted Product Store Sales")
    st.success(f"Estimated Sales Amount: ₹ {prediction:,.2f}")
