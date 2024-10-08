Before all to import the ml_toolkit and to import the torchinfo

Main points:

1. Download the helper functions from the github
2. Download the data
3. Restructure the data (for the model)
4. Create a DataSet and DataLoader
5. Getting a pre-trained model
6. Freeze the base layer and changing the classifier head
7. Train the model
8. View the model results
9. Save the model
10. Reload the saved model
11. Make prediction with the reloaded model

## Import the toolkit

import torch
import torchvision
from torch import nn
from torchvision import transforms
import os

# Try to get torchinfo, install it if it doesn't work

try:
from torchinfo import summary
except:
print("[INFO] Couldn't find torchinfo... installing it.")
!pip install -q torchinfo
from torchinfo import summary

# Try to import the apt_toolkit directory, download it from GitHub if it doesn't work

try:
from atp_toolkit import data_setup, engine
except: # Get the going_modular scripts
print("[INFO] Couldn't find ml_toolkit scripts... downloading them from GitHub.")
!git clone https://github.com/DanielSzakacs/atp_toolkit

<!--
To view the
%load_ext tensorboard
%tensorboard --logdir runs -->

NUM_WORKERS = os.cpu_count()

# Import tabnet

!pip install pytorch-tabnet

# Check if the player name is exist or not

prediction.checkSimilarNameInEncoder(encoder_path="./encoder",
encoder_name="Player2",
name_to_check="Duclos")

# Source of data

Odds: https://www.oddsportal.com/football/argentina/liga-profesional/
TabNetClassifier info: https://pypi.org/project/pytorch-tabnet/
Tennis Odds data: https://www.football-data.co.uk/englandm.php
Tennis Players data (GitHub): https://github.com/JeffSackmann/tennis_atp/blob/master/atp_matches_2023.csv
