
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ===============================
# Load data from Hugging Face
# ===============================
Xtrain_path = "hf://datasets/armakar123/my_Superkart/Xtrain.csv"
Xtest_path  = "hf://datasets/armakar123/my_Superkart/Xtest.csv"
ytrain_path = "hf://datasets/armakar123/my_Superkart/ytrain.csv"
ytest_path  = "hf://datasets/armakar123/my_Superkart/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest  = pd.read_csv(ytest_path).squeeze()

# ===============================
# Feature groups
# ===============================
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year'
]

categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Id',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]

# ===============================
# Preprocessing
# ===============================
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ===============================
# XGBoost Regressor
# ===============================
xgb_model = xgb.XGBRegressor(
    random_state=42,
    objective="reg:squarederror"
)

# ===============================
# Hyperparameter grid
# ===============================
param_grid = {
    "xgbregressor__n_estimators": [100, 200],
    "xgbregressor__max_depth": [3, 4, 5],
    "xgbregressor__learning_rate": [0.05, 0.1],
    "xgbregressor__subsample": [0.8, 1.0],
    "xgbregressor__colsample_bytree": [0.8, 1.0],
}

# ===============================
# Pipeline + GridSearch
# ===============================
model_pipeline = make_pipeline(preprocessor, xgb_model)

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# ===============================
# Evaluation
# ===============================
y_pred_train = best_model.predict(Xtrain)
y_pred_test  = best_model.predict(Xtest)

print(f"\nTrain RMSE: {mean_squared_error(ytrain, y_pred_train, squared=False):.2f}")
print(f"Test RMSE : {mean_squared_error(ytest, y_pred_test, squared=False):.2f}")
print(f"Test R²   : {r2_score(ytest, y_pred_test):.3f}")

# ===============================
# Save model
# ===============================
model_path = "best_superkart_model_v1.joblib"
joblib.dump(best_model, model_path)

# ===============================
# Upload to Hugging Face
# ===============================
repo_id = "armakar123/superkart-sales-model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("✅ Model uploaded successfully to Hugging Face")

