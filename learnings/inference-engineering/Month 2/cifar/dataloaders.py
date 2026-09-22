from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch

def build_dataloaders(batch_size, num_workers):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataroot = Path("../Datasets").resolve()
    print(dataroot)
    train_data = CIFAR10(root=dataroot, train=True, transform=transforms, download=False)
    test_data = CIFAR10(root=dataroot, train=False, transform=transforms, download=False)

    train_n = len(train_data.data)
    test_n = len(test_data.data)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    classes = ("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")

    return train_loader, test_loader, classes

def print_sample_data(dataloader, classes):
    n = len(dataloader.dataset)
    plt.figure(figsize=(6,6))
    idxes = torch.randint(0, n, size=(9,1))
    for i in range(9):
        plt.subplot(3,3,i+1)
        img = dataloader.dataset[idxes[i]]
        plt.imshow((img[0]/2+0.5).permute(1,2,0))
        plt.title(classes[img[1]])
        plt.axis("off")
    plt.show()
