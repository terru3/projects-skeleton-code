import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import os

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains:
        csv files with image_id and image_label
        real image directory
    """

    def __init__(self, image_dir = "../train_images", csv_dir = ".."):
        self.train_csv = pd.read_csv(csv_dir + "/train.csv")
        self.image_dir = image_dir

    def __getitem__(self, index):
        image_id, image_label = self.train_csv.iloc[index]

        # get the images and transform to tensor
        image  = Image.open(self.image_dir + image_id)
        image = image.resize((120, 120))
        image_tensor = transforms.ToTensor()(image)

        # get the labels
        label = image_label

        return image_tensor, label

    def __len__(self):
        return len(self.train_csv)
