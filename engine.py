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
                             name:str,
                             model,
                             X_train, 
                             X_test, 
                             y_train, 
                             y_test,
                             eval_name=["val"],
                             eval_metric=["accuracy"],
                             max_epochs:int=100,
                             patience:int=20,
                             batch_size:int=5024,
                             v_batch_size:int=128,
                             drop_last:bool=False
                             ):
  
  """
    Train the model. 
    X_train, X_test, y_train, y_test should be reacheable for the function

    Args: 
      eval_name=["val"],
      eval_metric=["accuracy"],
      max_epochs:int=100,
      patience:int=20,
      batch_size:int=5024,
      v_batch_size:int=128,
      drop_last:bool=False

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

  print(f"[INFO] Save model as `model_{name}.`")

  return model

# def train_with_writer(model: TabNetClassifier,
#                       train_dataloader: torch.utils.data.DataLoader,
#                       test_dataloader: torch.utils.data.DataLoader,
#                       optimizer: optim.Optimizer,
#                       loss_fn: nn.Module,
#                       epochs: int,
#                       device: torch.device,
#                       writer: SummaryWriter) -> Dict[str, List]:
#     """Trains and tests a TabNet model.

#     Passes a target TabNet model through train_step() and test_step()
#     functions for a number of epochs, training and testing the model
#     in the same epoch loop.

#     Calculates, prints and stores evaluation metrics throughout.

#     Stores metrics to specified writer log_dir if present.

#     Args:
#         model: A TabNet model to be trained and tested.
#         train_dataloader: A DataLoader instance for the model to be trained on.
#         test_dataloader: A DataLoader instance for the model to be tested on.
#         optimizer: An optimizer to help minimize the loss function.
#         loss_fn: A loss function to calculate loss on both datasets.
#         epochs: An integer indicating how many epochs to train for.
#         device: A target device to compute on (e.g. "cuda" or "cpu").
#         writer: A SummaryWriter() instance to log model results to.

#     Returns:
#         A dictionary of training and testing loss as well as training and
#         testing accuracy metrics. Each metric has a value in a list for
#         each epoch.
#     """
#     # Create empty results dictionary
#     results = {"train_loss": [],
#                "train_acc": [],
#                "test_loss": [],
#                "test_acc": []}

#     # Loop through training and testing steps for a number of epochs
#     for epoch in tqdm(range(epochs)):
#         train_loss, train_acc = train_step(model=model,
#                                            dataloader=train_dataloader,
#                                            loss_fn=loss_fn,
#                                            optimizer=optimizer,
#                                            device=device)
#         test_loss, test_acc = test_step(model=model,
#                                         dataloader=test_dataloader,
#                                         loss_fn=loss_fn,
#                                         device=device)

#         # Print out what's happening
#         print(
#             f"Epoch: {epoch+1} | "
#             f"train_loss: {train_loss:.4f} | "
#             f"train_acc: {train_acc:.4f} | "
#             f"test_loss: {test_loss:.4f} | "
#             f"test_acc: {test_acc:.4f}"
#         )

#         # Update results dictionary
#         results["train_loss"].append(train_loss)
#         results["train_acc"].append(train_acc)
#         results["test_loss"].append(test_loss)
#         results["test_acc"].append(test_acc)

#         # Log results to TensorBoard
#         if writer:
#             writer.add_scalars(main_tag="Loss",
#                                tag_scalar_dict={"train_loss": train_loss,
#                                                 "test_loss": test_loss},
#                                global_step=epoch)
#             writer.add_scalars(main_tag="Accuracy",
#                                tag_scalar_dict={"train_acc": train_acc,
#                                                 "test_acc": test_acc},
#                                global_step=epoch)
#             writer.flush()

#     return results


# def train_step(model: TabNetClassifier,
#                dataloader: torch.utils.data.DataLoader,
#                loss_fn: nn.Module,
#                optimizer: optim.Optimizer,
#                device: torch.device) -> Tuple[float, float]:
#     """Trains a TabNet model for a single epoch.

#     Turns a target TabNet model to training mode and then
#     runs through all of the required training steps (forward
#     pass, loss calculation, optimizer step).

#     Args:
#         model: A TabNet model to be trained.
#         dataloader: A DataLoader instance for the model to be trained on.
#         loss_fn: A loss function to minimize.
#         optimizer: An optimizer to help minimize the loss function.
#         device: A target device to compute on (e.g. "cuda" or "cpu").

#     Returns:
#         A tuple of training loss and training accuracy metrics.
#     """
#     model.train()

#     train_loss, train_acc = 0, 0

#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Forward pass
#         y_pred = model.predict(X)

#         # Calculate loss
#         loss = loss_fn(y_pred, y)
#         train_loss += loss.item()

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Calculate accuracy
#         y_pred_class = torch.argmax(y_pred, dim=1)
#         train_acc += (y_pred_class == y).sum().item() / len(y)

#     train_loss /= len(dataloader)
#     train_acc /= len(dataloader)
#     return train_loss, train_acc


# def test_step(model: TabNetClassifier,
#               dataloader: torch.utils.data.DataLoader,
#               loss_fn: nn.Module,
#               device: torch.device) -> Tuple[float, float]:
#     """Tests a TabNet model for a single epoch.

#     Turns a target TabNet model to "eval" mode and then performs
#     a forward pass on a testing dataset.

#     Args:
#         model: A TabNet model to be tested.
#         dataloader: A DataLoader instance for the model to be tested on.
#         loss_fn: A loss function to calculate loss on the test data.
#         device: A target device to compute on (e.g. "cuda" or "cpu").

#     Returns:
#         A tuple of testing loss and testing accuracy metrics.
#     """
#     model.eval()

#     test_loss, test_acc = 0, 0

#     with torch.no_grad():
#         for batch, (X, y) in enumerate(dataloader):
#             X, y = X.to(device), y.to(device)

#             # Forward pass
#             test_pred_logits = model.predict(X)

#             # Calculate loss
#             loss = loss_fn(test_pred_logits, y)
#             test_loss += loss.item()

#             # Calculate accuracy
#             test_pred_labels = torch.argmax(test_pred_logits, dim=1)
#             test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

#     test_loss /= len(dataloader)
#     test_acc /= len(dataloader)
#     return test_loss, test_acc
