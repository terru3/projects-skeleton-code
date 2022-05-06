from numpy import gradient
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork

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
    
    # Move model to GPU
    model = model.to(device)

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
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # can add weight_decay=0.01 as parameter for example—L2
    # in network: self.dropout = nn.Dropout(p=0.2) for dropout
    # inside forward(): x = self.dropout(x)

    loss_fn = nn.CrossEntropyLoss()
    
    my_transforms = transforms.Compose([
        transforms.GaussianBlur(kernel_size = (5, 9), sigma = (0.1, 0.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):

            # Backpropagation and gradient descent
            images, labels = batch

            # Move to GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Apply transforms
            images = my_transforms(images)

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

        # testing the evaluate function for now
        evaluate(val_loader, model, loss_fn, device)

def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    """
        
    # Set model to evaluate mode
    model.eval()
    
    # Pass model to GPU
    model = model.to(device)
   

    correct_num = 0
    total_num = 0
    for batch in val_loader:
        images, labels = batch

        # Passing to GPU
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images).argmax(axis=1)
  # output has row number of batch_size, and col number 1 (reduced from 5)
        correct_num += (predictions == labels).sum().item()
        total_num += len(labels)
        
    print("Accuracy: ", 100*(correct_num / total_num), "%")
    
    # Visualises sample images with their predicted and actual labels
    classes = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)",
               "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)",
               "Healthy"]
    for i in range(2):
        print("Prediction: ", classes[predictions[i]])
        print("Label: ", classes[labels[i]])
        plt.imshow(images[i].cpu().permute(1,2,0))
        plt.show()

