import os
import joblib
import pickle
import json
import torch
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from zipfile import ZipFile
import torch
import pandas as pd

def load_tabnet_model(model_zip_path):
    model = TabNetClassifier()
    model.load_model(model_zip_path)
    return model


# Load Scaler
def load_scaler(scaler_path: str):
  scaler = joblib.load(scaler_path)
  return scaler


def load_encoder(encoder_dir):
  for file_name in os.listdir(encoder_dir):
    print(f"file_name {file_name}")
    if file_name.endswith(".pkl"):
      with open(os.path.join(encoder_dir, file_name), 'rb') as f:
        encoder = pickle.load(f)
  return encoder


# Data procession
def preprocess_data(data, encoders, scaler):
  # Encode the categorical columns
  for col, encoder in encoders.items():
    if col in data.columns:
      data[col] = encoder.transform(data[col])

  # Scale the numerical columns
  data = scaler.transform(data)

  return data


def extract_date_feature(data_frame, date_column):
  data_frame[date_column] = pd.to_datetime(data_frame[date_column])
  data_frame[f"{date_column}_year"] = data_frame[date_column].dt.year.astype(np.int64)
  data_frame[f"{date_column}_month"] = data_frame[date_column].dt.month.astype(np.int64)
  data_frame[f"{date_column}_day"] = data_frame[date_column].dt.day.astype(np.int64)
  data_frame[f"{date_column}_dayofweek"] = data_frame[date_column].dt.dayofweek.astype(np.int64)
  return data_frame.drop(columns=[date_column])



def predict(model_path: str,
            scaler_path: str,
            encoder_path: str,
            raw_data: pd.DataFrame,
            exclude_features: list = [],
            data_column_name: str=None):
  print("[INFO] Load model")
  model = load_tabnet_model(model_zip_path=model_path)
  if data_column_name:
    print("[INFO] data_column_name is not null, new columns will be created")
    raw_data = extract_date_feature(raw_data, data_column_name)
  else:
    print("[INFO] data_column_name is null")
  print("[INFO] Scale numerical features")
  scaler = load_scaler(scaler_path)
  numerical_features = [col for col in raw_data.select_dtypes(include=["int64", "float64"]).columns if col not in exclude_features]
  print(f"[INFO] Numerical features: {numerical_features}")
  raw_data[numerical_features] = scaler.transform(raw_data[numerical_features])
  print(f"[INFO] Encode the object features")
  encoders = load_encoder(encoder_path)
  for column in raw_data.select_dtypes(include=['object']).columns:
    e = encoder[column]
    if e:
      print(f"Columns: {column}; Encoder: {e}")
      try:
        raw_data[column] = e.transform(raw_data[column])
      except ValueError as err :
        print(f"[Warning] Unsee label in column'{column}': {err}")



  print(f"[INFO] Dataframe to numpy")
  raw_data_np = raw_data.to_numpy()
  prediction = model.predict(raw_data_np)
  print(f"[PREDICTION]  {prediction}")