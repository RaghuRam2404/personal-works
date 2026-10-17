import torch
from torch.profiler import profile, ProfilerActivity, record_function
import torch.nn as nn

torch.cuda.is_available()

#### ALL HELPERS ####

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

device = 'cuda'
model = cifar_model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters())
scaler = torch.amp.GradScaler(device=device)

rand_count, rand_classes = 10000, 3
rand_input = torch.randn(size=(rand_count,3,32,32)).to(device)
rand_target = torch.randint(low=0, high=rand_classes, size=(rand_count,)).to(device)
amp_enabled = False

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities += [ProfilerActivity.CUDA]
model.train()

with record_function(name='one_full_loop'):
    with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
            with record_function(name='forward_pass'):
                logits = model(rand_input)
                print(logits.dtype)
            with record_function(name='calculate_loss'):
                loss = loss_fn(logits, rand_target)
                print(loss.dtype)
        
        if amp_enabled:
            with record_function(name='grad_scaler_backward'):
                scaler.scale(loss).backward()
            with record_function(name='update_gradients'):
                scaler.step(optimizer)
        else:
            with record_function(name='normal_backward'):
                loss.backward()
            with record_function(name='update_gradients'):
                optimizer.step()
        
        with record_function('zero_grad'):
            optimizer.zero_grad(set_to_none=True)  
              
    trace_file_name = "cuda_simple_fp32_trace.json" if not amp_enabled else "cuda_simple_amp_trace.json"
    prof.export_chrome_trace(trace_file_name)
