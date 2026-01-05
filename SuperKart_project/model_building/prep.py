# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/armakar123/my_Superkart/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['Product_Id'], inplace=True)

# Encoding the categorical column
label_encoder = LabelEncoder()
df['Product_Sugar_Content'] = label_encoder.fit_transform(df['Product_Sugar_Content'])
df['Product_Type'] = label_encoder.fit_transform(df['Product_Type'])
df['Store_Id'] = label_encoder.fit_transform(df['Store_Id'])
df['Store_Size'] = label_encoder.fit_transform(df['Store_Size'])
df['Store_Location_City_Type'] = label_encoder.fit_transform(df['Store_Location_City_Type'])
df['Store_Type'] = label_encoder.fit_transform(df['Store_Type'])



# Split into X (features) and y (target)
