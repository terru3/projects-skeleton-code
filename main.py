import os
import constants
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from starting_train.starting_train import starting_train

def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize train and validation datasets

    train_data = StartingDataset(train=True)
    val_data = StartingDataset()

    indices = list(range(len(train_data)))
    np.random.shuffle(indices)
    train_size = int(0.8 * len(train_data))
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_dataset = torch.utils.data.SubsetRandomSampler(train_idx)
    val_dataset = torch.utils.data.SubsetRandomSampler(val_idx)
    # To-do, use WeightedRandomSampler instead to sample more pictures of the rarer classes

    # Initialize, train and evaluate model
    model = StartingNetwork()
    starting_train(
        train_data=train_data,
        val_data=val_data,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
        device=device,
    )

if __name__ == "__main__":
    main()
