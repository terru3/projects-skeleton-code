from numpy import gradient
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from StartingDataset import StartingDataset
from StartingNetwork import StartingNetwork

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """
    # Set model to train mode
    model.train()

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):

            # Backpropagation and gradient descent
            images, labels = batch

            # Move to GPU
            images.to(device)
            labels.to(device)

            # Forward propagation
            outputs = model(images)

            # backward propagation, and gradient descent using our optimizer
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Periodically evaluate our model + log to Tensorboard
            """""
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                evaluate(val_loader, model, loss_fn, device)
            step += 1
            """""

        print('Epoch: ', epoch + 1, 'Loss: ', loss.item())  # print loss of the last batch for each epoch


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    """

    # Set model to evaluate mode
    model.eval()

    correct_num = 0
    for batch in val_loader:
        images, labels = batch

        # Passing to GPU
        images.to(device)
        labels.to(device)

        predictions = model(images).argmax(axis=1)
  # output has row number of batch_size, and col number 1 (reduced from 5)
        correct_num += (predictions == labels).sum().item()

    print("Accuracy: ", 100*(correct_num / len(labels)), "%")