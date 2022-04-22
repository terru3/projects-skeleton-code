import matplotlib.pyplot as plt
import torch
from StartingDataset import StartingDataSet

train = StartingDataset()
print(train[0])
print(len(train))

train_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
train_iter = iter(train_loader)
batch_images, batch_labels = next(train_iter)
plt.imshow(batch_images[0].permute(1,2,0))
plt.show()
