import torch
from torch import nn

class cfar(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #input size = (3, 32, 32)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 6, 5), # output size: (6, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # output size: (6, 14, 14)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # output size: (16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2,2) # output size: (16, 5, 5)
        )
        self.forward_block = nn.Sequential(
            nn.Flatten(), # output size: (n, 16*5*5=400)
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84,10) #logits
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.forward_block(x)
        return x
    
def get_model(device):
    return cfar().to(device=device)