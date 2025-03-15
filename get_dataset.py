import pandas as pd
import numpy as np
import matplotlib as plt


df = pd.read_csv("dataset/bank-full.csv", sep=';')

# Convert Categorical Columns to numeric values
categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]
for col in categorical_columns:
    df[col] = pd.Categorical(df[col]).codes

X = df.drop(columns=["y"]).to_numpy() 
y = df["y"].to_numpy()