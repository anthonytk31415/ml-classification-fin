import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import matplotlib as plt


# df = pd.read_csv("dataset/bank-full.csv", sep=';')

# # Convert Categorical Columns to numeric values
# categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]
# for col in categorical_columns:
#     df[col] = pd.Categorical(df[col]).codes

# X = df.drop(columns=["y"]).to_numpy() 
# y = df["y"].to_numpy()



# Load the dataset
input_df = pd.read_csv("dataset/bank-full.csv", sep=';')

# Print the first few rows to understand the data
print("Dataset preview:")
print(input_df.head())
print("\nDataset shape:", input_df.shape)

# Separate features and target
target_column = 'y'
y = (input_df[target_column] == 'yes').astype(int)
X = input_df.drop(target_column, axis=1)

# Identify numerical and categorical columns
categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
numerical_columns = [col for col in X.columns if col not in categorical_columns]

print("\nCategorical columns:", categorical_columns)
print("\nNumerical columns:", numerical_columns)

# Create a preprocessing pipeline with one-hot encoding and normalization
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ])

# Fit and transform the data
X_transformed = preprocessor.fit_transform(X)

# Get the feature names after one-hot encoding
ohe_feature_names = []
for i, encoder in enumerate(preprocessor.transformers_):
    if encoder[0] == 'cat':
        encoder_obj = encoder[1]
        feature_names = encoder_obj.get_feature_names_out(categorical_columns)
        ohe_feature_names.extend(feature_names)
    else:
        ohe_feature_names.extend(numerical_columns)

print("\nFeature names after one-hot encoding:")
for i, name in enumerate(ohe_feature_names):
    print(f"{i}: {name}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.toarray() if hasattr(X_train, 'toarray') else X_train)
X_test_tensor = torch.FloatTensor(X_test.toarray() if hasattr(X_test, 'toarray') else X_test)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)