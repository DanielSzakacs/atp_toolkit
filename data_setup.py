
import requests
import torch 
import os 
import pandas as pd
from torchvision import datasets, transforms
from pathlib import Path

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
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

