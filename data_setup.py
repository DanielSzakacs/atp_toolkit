
import requests
import torch 
import os 
import pandas as pd
from torchvision import datasets, transforms
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import pickle

import numpy as np


NUM_WORKERS = os.cpu_count()

def download_tennis_atp_match_stat_github(year: int,
                                          target_path: str = "github_atp_stat"):
  """
    Download Tennis match data from github https://github.com/JeffSackmann/tennis_atp/tree/master 
    Github repo url: https://github.com/JeffSackmann/tennis_atp/tree/master

    CSV files will be downloaded from here: https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv

    Args: 
      year: integer - all the files in the github repozitory is named after a year (2023 or 2022 or ...)

  """

  # Create local directory
  data_path = Path(target_path)

  if data_path.is_dir():
    print(f"firectory exists")
  else:
    print(f"Directory do not exist ... will be created")
    data_path.mkdir(parents=True, exist_ok=True)

  # Download data
  with open(data_path / f"{year}.csv", "wb") as f:
    request = requests.get(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv")
    print("Data download")
    f.write(request.content)

def download_tennis_data_co_uk_file_github(year: int,
                                          target_path: str = "tennis.co.uk_atp_stat"):
  """
    Source of the data is: http://www.tennis-data.co.uk/alldata.php
    Github repo url: https://github.com/DanielSzakacs/atp_data

    CSV files will be downloaded from here: https://raw.githubusercontent.com/DanielSzakacs/atp_data/main/2021.csv

    Args: 
      year: integer - all the files in the github repozitory is named after a year (2023 or 2022 or ...)

  """

  # Create local directory
  data_path = Path(target_path)

  if data_path.is_dir():
    print(f"firectory exists")
  else:
    print(f"Directory do not exist ... will be created")
    data_path.mkdir(parents=True, exist_ok=True)

  # Download data
  with open(data_path / f"{year}.csv", "wb") as f:
    request = requests.get(f"https://raw.githubusercontent.com/DanielSzakacs/atp_data/main/{year}.csv")
    print("Data download")
    f.write(request.content)


def show_csv_table(name: str, 
                   soure_path: str = "github_atp_stat"):
  """ Display csv file table, by using panda.read_csv 
  
    Args: 
      name: String - name of the file
      source_path: String - where we keep the data. By default it is from the github_atp_stat
  """
  data_path = Path(soure_path) / f"{name}.csv"
  if data_path.exists():
    print(f"Load {name}.csv file")
    return pd.read_csv(data_path)
  else:
    print(f"File {name}.csv do not exist")
    return None

def marge_data_dir_files(folderPath: str = "github_atp_stat"):
  """
    Will create a single dataframe from all the csv files which is located inside the /github_atp_stat folder.
    Marge happanes based on there functions.

    Make sure that only similar csv files are in the github_atp_stat folder
  """
  data_path = Path(folderPath)
  if data_path.exists():
    print(f"Load file from {folderPath}")
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    dataframes = []

    # Loop through each CSV file and read it into a DataFrame
    for csv_file in csv_files:
      print(f"Load file {csv_file}")
      file_path = os.path.join(data_path, csv_file)

      # Read the CSV file into a pandas DataFrame
      df = pd.read_csv(file_path)

      # Append the DataFrame to the list
      dataframes.append(df)
  else:
    print(f"Can not load csv files")
    return None

  # Concatenate all DataFrames into one
  data = pd.concat(dataframes, ignore_index=True)
  return data


def form_dataframe_functions(df):
  """ 
    This functinos will remove some of the columns and keep only these: 
    [Location, Tournament, Court, Surface, Player1, Player2, Odds1, Odds2, Odds3, Target]
  """
  data = []

  for index, row in df.iterrows():
    data.append([ row["Location"],
                 row["Tournament"],
                 row["Date"], 
                  row["Court"],
                  row["Surface"],
                  row["Winner"], row["Loser"],
                  row["B365W"], row["B365L"],
                  row["PSW"], row["PSL"],
                  row["AvgW"], row["AvgL"],
                   row["MaxW"], row["MaxL"], 1])

    data.append([row["Location"],
                row["Tournament"],
                 row["Date"],
                 row["Court"],
                 row["Surface"],
                 row["Loser"], row["Winner"],
                 row["B365L"], row["B365W"],
                 row["PSL"], row["PSW"],
                 row["AvgL"], row["AvgW"], 
                 row["MaxL"], row["MaxW"], 0])

  # Create a new DataFrame
  new_df = pd.DataFrame(data, columns=["Location", 
                                       "Tourney_name", 
                                       "Date",
                                       "Court", 
                                       "Surface",
                                       "Player1", "Player2",
                                       "B3651", "B3652",
                                       "PS1", "PS2",
                                       "Avg1", "Avg2",
                                       "Max1", "Max2",
                                       "Winner"])
  print(f"Createing dataframe. Original length: {len(df)} | New length {len(new_df)}")

  return new_df


def form_dataframe_functions_oods(df):
  """ 
    This functinos will remove ALL the columns which are not odds data: 
    [Odds1, Odds2, Odds3, Target]

    Odds data 
    Odds1 = B365
    Odds2 = Pinnacle
    Odds3 = Average

    You can find these odds data for the next match in here: https://www.oddsportal.com/football/argentina/liga-profesional/ind-rivadavia-independiente-vH4fu7lB/#1X2;2
  """
  data = []

  for index, row in df.iterrows():
    data.append([row["B365W"], row["B365L"],
                  row["PSW"], row["PSL"],
                  row["AvgW"], row["AvgL"],
                   row["MaxW"], row["MaxL"], 1])

    data.append([row["B365L"], row["B365W"],
                 row["PSL"], row["PSW"],
                 row["AvgL"], row["AvgW"],
                  row["MaxL"], row["MaxW"], 0])

  # Create a new DataFrame
  new_df = pd.DataFrame(data, columns=["B3651", "B3652",
                                       "PS1", "PS2",
                                       "Avg1", "Avg2",
                                       "Max1", "Max2",
                                       "Winner"])
  print(f"Createing dataframe. Original length: {len(df)} | New length {len(new_df)}")

  return new_df


def form_dataframe_v2(df):
  """ 
    This functinos will remove some of the columns and keep only these: 
    [Date, Player1, Player2, Odds1, Odds2, Odds3, Target]
  """
  data = []

  for index, row in df.iterrows():
    data.append([ row["Date"], 
                  row["Winner"], row["Loser"],
                  row["B365W"], row["B365L"],
                  row["PSW"], row["PSL"],
                  row["AvgW"], row["AvgL"],
                   row["MaxW"], row["MaxL"], 1])

    data.append([row["Date"],
                 row["Loser"], row["Winner"],
                 row["B365L"], row["B365W"],
                 row["PSL"], row["PSW"],
                 row["AvgL"], row["AvgW"], 
                 row["MaxL"], row["MaxW"], 0])

  # Create a new DataFrame
  new_df = pd.DataFrame(data, columns=["Date",
                                       "Player1", "Player2",
                                       "B3651", "B3652",
                                       "PS1", "PS2",
                                       "Avg1", "Avg2",
                                       "Max1", "Max2",
                                       "Winner"])
  print(f"Createing dataframe. Original length: {len(df)} | New length {len(new_df)}")

  return new_df



def extract_date_feature(df, date_column):
  df[date_column] = pd.to_datetime(df[date_column])
  df[f"{date_column}_year"] = df[date_column].dt.year.astype(np.int64)
  df[f"{date_column}_month"] = df[date_column].dt.month.astype(np.int64)
  df[f"{date_column}_day"] = df[date_column].dt.day.astype(np.int64)
  df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek.astype(np.int64)


  return df.drop(columns=[date_column])


# Split, form date, skale numerical data, encode object type of features
def prepare_data_with_data(data: pd.DataFrame,
                 target_column: str,
                 test_size: float,
                 date_column: str,
                 save_encoder: bool=False,
                 fillout_numerical_with_mean: bool = False):
    """
    Removes all rows which contains NaN
    Encode the objects
    Saves the encoder if params true
    Split data to train and test

    Args:
      data: pd.DataFrame
      target_column: str - name of the target
      test_size: float,
      save_encoder: bool

    Returns:
      X_train, X_test, y_train, y_test
      encoder
    """

    print(f"[INFO] Input data length: {len(data)}")
    # Shuffle the data rows
    data = data.sample(frac=1).reset_index(drop=True)

    # Drop NaN rows
    if fillout_numerical_with_mean:
      print(f"[INFO] Fillout NaN with mean")
      numerical_features = [col for col in data.select_dtypes(include=["int64", "float64"]).columns if col not in ["Winner"]]
      data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())
    
    clear_df = data.dropna()
    print(f"[INFO] {len(data) - len(clear_df)} rows removed because of NaN features")

    # Process date columns
    if(date_column):
      clear_df = extract_date_feature(clear_df, date_column)
      print(f"[INFO] Processed data columns: {date_column}")

    # Split data to X and y
    X = clear_df.drop(columns=[target_column])
    y = clear_df[target_column]
    print(f"[INFO] Split to X and y")

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print(f"[INFO] Data split to train and test data")

    # Scale numerical data
    scaler = StandardScaler()
    # Get the numerical features (excluding extracted date columns)
    exclude_features = []
    numerical_features = [col for col in X_train.select_dtypes(include=["int64", "float64"]).columns if col not in exclude_features]
    # X_train.select_dtypes(include=["int64", "float64"]).columns

    # Fit the scaler on the training data and transform the training data
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    # Transform the test data
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    print(f"[INFO] Scaler: {len(numerical_features)} number of features have been scaled (Both train and test data)")

    # Save scaler
    scaler_path = Path("scaler/scaler.joblib")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Scaler saved to {scaler_path}")

    # Encode X features if type object
    encoder = {}
    for column in X.select_dtypes(include=['object']).columns:
        # Convert all values to strings to ensure uniformity
        X_train[column] = X_train[column].astype(str)
        X_test[column] = X_test[column].astype(str)
        le = LabelEncoder()
        X_train[column] = le.fit_transform(X_train[column])
        try:
            X_test[column] = le.transform(X_test[column])
        except ValueError as e:
            # Log the error and assign a default value (e.g., -1) to unseen labels
            print(f"[WARNING] Unseen label in column '{column}': {e}")
            unseen_labels = set(X_test[column].unique()) - set(le.classes_)
            unseen_label_map = {label: -1 for label in unseen_labels}
            X_test[column] = X_test[column].map(lambda x: le.transform([x])[0] if x in le.classes_ else unseen_label_map[x])
        encoder[column] = le
    print(f"[INFO] LabelEncoder logic finished. {len(encoder)} number of columns have been encoded")

    # Save the encoder if save_encoder is True
    if save_encoder:
        encoder_path = Path("encoder")
        print(f"[INFO] Saving encoders to folder '{encoder_path}'.")
        encoder_path.mkdir(parents=True, exist_ok=True)

        # Save each encoder to the folder using pickle
        for key, value in encoder.items():
            path = encoder_path / f"{key}_encoder.pkl"
            with open(path, 'wb') as f:
                pickle.dump(value, f)  # Save using pickle
            print(f"[INFO] Saved encoder '{key}_encoder.pkl' to {path}")
    

    # Return
    return X_train, X_test, y_train, y_test, encoder