
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
                #  row["Date"], 
                  row["Court"],
                  row["Surface"],
                  row["Winner"], row["Loser"],
                  row["B365W"], row["B365L"],
                  row["PSW"], row["PSL"],
                  row["AvgW"], row["AvgL"], 1])

    data.append([row["Location"],
                row["Tournament"],
                #  row["Date"],
                 row["Court"],
                 row["Surface"],
                 row["Loser"], row["Winner"],
                 row["B365L"], row["B365W"],
                 row["PSL"], row["PSW"],
                 row["AvgL"], row["AvgW"], 0])

  # Create a new DataFrame
  new_df = pd.DataFrame(data, columns=["Location", 
                                       "Tourney_name", 
                                       "Court", 
                                       "Surface",
                                       "Player1", "Player2",
                                       "B3651", "B3652",
                                       "PS1", "PS2",
                                       "Avg1", "Avg2",
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
                  row["AvgW"], row["AvgL"], 1])

    data.append([row["B365L"], row["B365W"],
                 row["PSL"], row["PSW"],
                 row["AvgL"], row["AvgW"], 0])

  # Create a new DataFrame
  new_df = pd.DataFrame(data, columns=["B3651", "B3652",
                                       "PS1", "PS2",
                                       "Avg1", "Avg2",
                                       "Winner"])
  print(f"Createing dataframe. Original length: {len(df)} | New length {len(new_df)}")

  return new_df


def create_dataloaders(
    data: pd.DataFrame,
    target_column: str,
    batch_size: int,
    num_workers: int = 4,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Creates training and testing DataLoaders for tabular data.

    Args:
        data: Pandas DataFrame containing the entire dataset.
        target_column: Name of the target column.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed used by the random number generator for train/test split.

    Returns:
        A tuple of (train_dataloader, test_dataloader, feature_names, class_names).
        Where class_names is a list of the target classes.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical features and target
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = le.classes_.tolist()
    else:
        if y.nunique() < 20:  # Arbitrary threshold to assume it's a classification task
            class_names = y.unique().tolist()
            class_names.sort()  # Ensure class names are sorted

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feature_names = X.columns.tolist()

    return train_dataloader, test_dataloader, feature_names, class_names

def create_dataloaders_with_encoder(
    data: pd.DataFrame,
    target_column: str,
    batch_size: int,
    num_workers: int = 4,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Creates training and testing DataLoaders for tabular data.

    Args:
        data: Pandas DataFrame containing the entire dataset.
        target_column: Name of the target column.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed used by the random number generator for train/test split.

    Returns:
        A tuple of (train_dataloader, test_dataloader, feature_names, class_names, encoders).
        Where class_names is a list of the target classes and encoders is a dictionary of LabelEncoders.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    encoders = {}
    
    # Encode categorical features
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        encoders[column] = le  # Store the encoder
    
    class_names = None
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = le.classes_.tolist()
        encoders[target_column] = le  # Store the encoder
    else:
        if y.nunique() < 20:  # Arbitrary threshold to assume it's a classification task
            class_names = y.unique().tolist()
            class_names.sort()  # Ensure class names are sorted
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feature_names = X.columns.tolist()

    return train_dataloader, test_dataloader, feature_names, class_names, encoders

def prepare_data(data: pd.DataFrame,
                 target_column: str,
                 test_size: int,
                 save_encoder: bool=False):
  """
    Removes all rows which contains NaN
    Encode the objects
    Saves the encoder if params true
    Split data to train and test

    Args: 
      data: pd.DataFrame
      target_column: str - name of the target
      test_size: int,
      save_encoder: bool

    Returns: 
      X_train, X_test, y_train, y_test
      encoder
  """

  print(f"[INFO] Input data length: {len(data)}")
  # Drop NaN rows
  clear_df = data.dropna()
  print(f"[INFO] {len(data) - len(clear_df)} rows removed because of NaN features")

  # Split data to X and y 
  X = clear_df.drop(columns=[target_column])
  y = clear_df[target_column]
  print(f"[INFO] DataFrame has been splid to X and y")

  # Encode X features if type object 
  encoder = {}
  for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    encoder[column] = le
  print(f"[INFO] LabelEncoder logic finished. {len(encoder)} number of columns has been numerized")

  # Save the encoder if save_encoder == true
  if save_encoder :
      encoder_path = Path("encoder")
                   
      print(f"[INFO] Save encoders to folder")
      # Check if the folder exist
      if encoder_path.is_dir():
        print(f"[INFO] Encoder folder exist")
      else:
        print(f"[INFO] Creating encoders folder ...")
        encoder_path.mkdir(parents=True, exist_ok=True)
      # Save endoder to the folder
      for key, value in encoder.items():
        print(f"[INFO] Save encoder by the name {key}_encoder.pkl")
        path = encoder_path / f"{key}_encoder.pkl"
        joblib.dump(value, path)

  # Split to train and test data 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
  print(f"[INFO] Data splited to train and test data")

  # Return
  return X_train, X_test, y_train, y_test, encoder