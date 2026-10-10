import torch
from torch.profiler import profile, ProfilerActivity, record_function
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from pathlib import Path

#### GLOBAL VARIABLES ####

CIFAR10_PATH = "./" #"../Datasets/"
CAN_DOWNLOAD = True
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#### ALL HELPERS ####

tranforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.RandomCrop(size=(32,32), padding=4),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def plot_data(data):
    plt.figure(figsize=(5,5))
    for i in range(9):
        plt.subplot(3,3,i+1)
        
        #for pil image input, before the transform
        #plt.imshow(data[i][0]) 
        
        #after the transform
        image = data[i][0]
        image = (image+1.0)/2.0 # unnormalize
        image = image.permute(1,2,0) # channel change
        plt.imshow(image) 
        
        plt.axis("off")
        plt.title(CIFAR10_CLASSES[data.targets[i]])
    plt.show()

class cifar_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # input shape 3x32x32
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), # 64 x 32 x 32
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # 64 x 32 x 32
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 64 x 16 x 16
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # 128 x 16 x 16
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # 128 x 16 x 16
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 128 x 8 x 8
        )
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.linear_block = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.adaptive_avg_pool(x)
        x = self.linear_block(x)
        return x
    

#### MAIN CODE ###

train = CIFAR10(root=CIFAR10_PATH, train=True, download=CAN_DOWNLOAD, transform=tranforms)
test = CIFAR10(root=CIFAR10_PATH, train=False, download=CAN_DOWNLOAD, transform=tranforms)
train_n = len(train.targets)
print(min(train.targets), max(train.targets), len(CIFAR10_CLASSES))

idx=0
#img = TF.pil_to_tensor(train[idx][0]) #will work before transform
#img.min(), img.max(), img.shape # 0.0, 255.0, 3x32x32
img = train[0][0] #after the tranform
img.min(), img.max(), img.shape # -1.0 1.0 3x32x32


plot_data(train)

model = cifar_model()
model.eval()

rand_count, rand_classes = 5, 3
rand_input = torch.randn(size=(rand_count,3,32,32))
rand_target = torch.randint(low=0, high=rand_classes, size=(rand_count,))
with torch.no_grad():
    logits = model(rand_input)
print(logits.shape)

total_params = sum(p.numel() for p in model.parameters()) #262218
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) #262218

loss_fn = nn.CrossEntropyLoss()
loss_fn(logits, rand_target)

### Training Loop ###

def train_one_loop(model, data_loader, optim, loss_fn, device):
    #with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
    tlosses = []
    start, end, elapsed_ms = None, None, None
    if device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter_ns()
        
    for batch_no, (x,y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        batch_loss = loss_fn(logits, y)
        tlosses.append(batch_loss.item())
        
        batch_loss.backward()
        optim.step()
        
        optim.zero_grad(set_to_none=True)
        
    if device == 'cuda':
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        end = time.perf_counter_ns()
        elapsed_ms = (end - start)/(1000**2)
    #prof.export_chrome_trace("single_batch_training_loop.json")
    
    return tlosses, elapsed_ms

def training_loop(model, data_loader, epochs, epochs_trained, optim, loss_fn, device, store_checkpt):
    if epochs_trained is None:
        epochs_trained = 0
    losses = []
    for epoch in range(epochs):
        tlosses, elapsed_ms = train_one_loop(model, data_loader, optim, loss_fn, device)
        loss = (sum(tlosses)/len(tlosses))
        losses.append(loss)
        
        print(f"Epoch: {(epochs_trained+epoch+1):3}/{epochs_trained+epochs:2} and loss:{loss:.5f} in {elapsed_ms} ms time")
        if store_checkpt is not None and store_checkpt == True:
            store_checkpoint(model, optim, epochs_trained+epoch+1, checkpt_path)
            
    return losses

def store_checkpoint(model, optim, epochs, file):
    check_point = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epochs_done' : epochs
    }
    torch.save(check_point, file)
    
def load_checkpoint(file, device, model, optim):
    check_point = torch.load(file, map_location=device, weights_only=True)
    model_dict = check_point['model']
    optim_dict = check_point['optim']
    epochs_done = check_point['epochs_done']
    model.load_state_dict(model_dict)
    
    if optim is not None:
        optim.load_state_dict(optim_dict)
    
    print(f"Checkpoint loaded. Total epochs done till now: {epochs_done}")
    return model_dict, optim_dict, epochs_done

FRESH_TRAINING = True
checkpt_path = "./cifarfp32_1.pth"

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = cifar_model().to(device)
optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
epochs_trained = None
model.train()

if not FRESH_TRAINING:
    model_dict, optim_dict, epochs_done = load_checkpoint(checkpt_path, device, model, optim)
    epochs_trained = epochs_done

epochs_to_run_now = 15
batch_size = 1024
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)


activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities += [ProfilerActivity.CUDA]
#with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
training_loop(model, train_loader, epochs_to_run_now, epochs_trained, optim, loss_fn, device, True)
#prof.export_chrome_trace("cuda_cifar_fp32_trace.json")

### Test accuracy logic ###

model = cifar_model().to(device)
load_checkpoint(checkpt_path, device, model, None)

test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
test_n = test.data.shape[0]

model.eval()

correct = 0
with torch.no_grad():
    for batch_no, (tx, ty) in enumerate(test_loader):
        tx, ty = tx.to(device), ty.to(device)
        logits = model(tx)
        ypred = torch.argmax(logits, dim=1)
        correct += torch.eq(ypred, ty).int().sum().item()

print(f"Accuracy: {(correct/test_n):.2f} with '{correct}' correct predictions out of '{test_n}'")