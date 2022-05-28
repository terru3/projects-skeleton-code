import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import os

my_transforms = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180))
        ])

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains:
        csv files with image_id and image_label
        real image directory
    """

    def __init__(self, image_dir = "/train_images/",
                 csv_dir = "", train=False):
        self.train_csv = pd.read_csv(csv_dir + "/train.csv")
        self.image_dir = image_dir
        self.train = train

    def __getitem__(self, index):
        image_id, image_label = self.train_csv.iloc[index]

        # get the images and transform to tensor
        image  = Image.open(self.image_dir + image_id)
        image = image.resize((224, 224))
        image_tensor = transforms.ToTensor()(image)

        # normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)

        # transformations
        if (self.train):
            image_tensor = my_transforms(image_tensor)

        # TO-DO: Only apply transformations if class != 3 (the most common one)

        # get the labels
        label = image_label

        return image_tensor, label

    def __len__(self):
        return len(self.train_csv)
