import torch
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

### Download data

mnist_train = FashionMNIST(
    root = "data",
    train=True,
    download=True,
    transform=ToTensor()
)

mnist_test = FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

target_classes = mnist_train.classes
print(target_classes)
classes_with_index = mnist_train.class_to_idx
print(classes_with_index)

train_shape = mnist_train.data.shape # [m,dim1,dim2]
m = train_shape[0]
print("m: {}".format(m))

raw_folder, processed_folder = mnist_train.raw_folder, mnist_train.processed_folder
print("Raw Folder: {}, Processed Folder: {}".format(raw_folder, processed_folder))

data, label = mnist_train[0]
print(data.shape, label)

### Visualizing

figure = plt.figure(figsize=(8,8))
rows, cols = 3, 3
for i in range(1, rows*cols+1):
    idx = np.random.randint(0, len(mnist_train))
    img, target = mnist_train[idx]
    figure.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(target_classes[target])
    plt.axis("off")
plt.show()

