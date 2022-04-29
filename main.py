import os
import constants
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from StartingDataset import StartingDataset
from StartingNetwork import StartingNetwork
from starting_train import starting_train


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize train and validation datasets
    dataset = StartingDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # this may not yield balanced number of images from each class, however

    # Initialize, train and evaluate model
    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
        device=device,
    )

# Visualises sample images with their predicted and actual labels
classes = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)",
           "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)",
           "Healthy"]
# for i in range(2):
    # print("Prediction: ") #smth like classes[predictions[i]]
    # print("Label: ") #smth like classes[labels[i]]
    # smth like plt.imshow (images[i].permute(1,2,0)) (may need .cpu())
    # plt.show()

if __name__ == "__main__":
    main()