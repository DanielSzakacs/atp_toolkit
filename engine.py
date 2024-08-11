"""
Contains functions for training and testing a PyTorch model.
"""

from typing import Tuple, Dict, List
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pytorch_tabnet.tab_model import TabNetClassifier
import os
from sklearn.metrics import accuracy_score


def create_tab_net_classifier(optimizer=torch.optim.Adam,
                              lr: int=0.2,
                              step_size: int=10,
                              gamma: int=0.9,
                              scheduler_fn=torch.optim.lr_scheduler.StepLR,
                              mask_type: str="sparsemax"
                              ):
  """
    Create basic TabNetClassifier

    Args:
      optimizer=torch.optim.Adam,
      lr: int=0.2,
      step_size: int=10,
      gamma: int=0.9,
      scheduler_fn=torch.optim.lr_scheduler.StepLR,
      mask_type: str="sparsemax"

    Return: 
      TabNetClassifier
  """
  print(f"[INFO] Make sure you install TabNetClasifier. More info -> Readme.txt")
  clf = TabNetClassifier(
    optimizer_fn=optimizer,
    optimizer_params=dict(lr=lr),
    scheduler_params={"step_size":step_size, # how to use learning rate scheduler
                      "gamma":gamma},
    scheduler_fn=scheduler_fn,
    mask_type=mask_type
  )
  print(f"[INFO] Model created")
  return clf


def train_tab_net_classifier(
                             name: str,
                             model,
                             X_train, 
                             X_test, 
                             y_train, 
                             y_test,
                             eval_name=["val"],
                             eval_metric=["accuracy"],
                             max_epochs: int=100,
                             patience: int=20,
                             batch_size: int=5024,
                             v_batch_size: int=128,
                             drop_last: bool=False
                             ):
  
    """
    Train the model. 
    X_train, X_test, y_train, y_test should be reachable for the function

    Args: 
      eval_name=["val"],
      eval_metric=["accuracy"],
      max_epochs: int=100,
      patience: int=20,
      batch_size: int=5024,
      v_batch_size: int=128,
      drop_last: bool=False

    Return: 
      Trained model
    """
    NUM_WORKERS = os.cpu_count()

    # Convert your data to NumPy arrays
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values
    y_test_np = y_test.values

    model.fit(X_train_np, y_train_np,
              eval_set=[(X_test_np, y_test_np)],
              eval_name=eval_name,
              eval_metric=eval_metric,
              max_epochs=max_epochs,
              patience=patience,
              batch_size=batch_size,
              virtual_batch_size=v_batch_size,
              num_workers=NUM_WORKERS,
              drop_last=drop_last)  # Set the learning rate here

    y_pred = model.predict(X_test_np)

    accuracy = accuracy_score(y_test_np, y_pred)
    print(f"Test accuracy {accuracy * 100:.2f}%")

    print(f"[ARGS] Length of the X_train: {len(X_train)}")
    print(f"[ARGS] Eval_name: {eval_name}")
    print(f"[ARGS] Eval_metric: {eval_metric}")
    print(f"[ARGS] Max_epochs: {max_epochs}")
    print(f"[ARGS] Patience: {patience}")
    print(f"[ARGS] Batch_size: {batch_size}")
    print(f"[ARGS] V_batch_size: {v_batch_size}")
    print(f"[ARGS] Drop_last: {drop_last}")

    # Ensure the /models directory exists
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(model_dir, f"model_{name}_{accuracy * 100:.2f}")
    model.save_model(model_path)
    # torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved as {model_path}")

    return model


def load_tab_net_classifier(path:str):
  clf = TabNetClassifier()
  clf.load_model(path)
  return clf 