import torch, time, gc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Setting torch device as {device}")
torch.set_default_device(device)

start_time = None

def start_timer():
    global start_time
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
    start_time = time.time()
    
def end_timer(msg):
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"{msg}")
    print(f"Total execution time {(end_time-start_time):.3f}")
    if device == "cuda":
        print(f"Max memory used by tensors = {torch.cuda.max_memory_allocated()} bytes")

def build_data(batch_size, in_size, out_size, num_batches):
    x = [torch.randn(size=(batch_size, in_size)) for _ in range(num_batches)]
    y = [torch.randn(size=(batch_size, out_size)) for _ in range(num_batches)]
    return x, y

def build_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers-1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers))

batch_size = 512
in_size = 4096
out_size = 4096
num_batches = 50
num_layers = 3
epochs = 5
x,y = build_data(batch_size, in_size, out_size, num_batches)
model = build_model(in_size, out_size, num_layers)
loss = torch.nn.MSELoss()

len(x), x[0].shape

## DEFAULT PRECISION IN CPU
device = "cpu"
model = build_model(in_size, out_size, num_layers)
loss_fn = torch.nn.MSELoss()

opt = torch.optim.SGD(params=model.parameters(), lr=0.001)
start_timer()
for epoch in range(epochs): #each epoch
    for tx,ty in zip(x,y): #each batch
        logits = model(tx)
        loss = loss_fn(logits, ty)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
end_timer("Default precision in CPU")

## With Autograd in CPU broken down
device = "cpu"
model = build_model(in_size, out_size, num_layers)
loss_fn = torch.nn.MSELoss()

tx, ty = x[1], y[1]
with torch.autocast(device_type=device, dtype=torch.float16):
    logits = model(tx)

with torch.autocast(device_type=device, dtype=torch.float16):
    loss = loss_fn(logits, ty)

start = time.perf_counter()
loss.backward() #slow because of fp16 based matmul logic in the laptop
end = time.perf_counter()
print(f"total-time: {end-start:.6f}s") #about 42seconds

logits.dtype
loss.dtype

opt.step()

opt.zero_grad(set_to_none=True)

## With Autograd in CPU (Don't run the below)
"""
device = "cpu"
model = build_model(in_size, out_size, num_layers)
loss_fn = torch.nn.MSELoss()

opt = torch.optim.SGD(params=model.parameters(), lr=0.001)
start_timer()
for epoch in range(1): #each epoch
    for tx,ty in zip(x,y): #each batch
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(tx)
            assert logits.dtype is torch.float16
            loss = loss_fn(logits, ty)
            assert loss.dtype is torch.float32
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
end_timer("With autograd to fp16 for 5 epochs in CPU")
"""

## With autograd and gradscaler in CPU/GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(in_size, out_size, num_layers).to(device)
loss_fn = torch.nn.MSELoss().to(device)

opt = torch.optim.SGD(params=model.parameters(), lr=0.001)
grad_scaler = torch.amp.GradScaler(device=device)
use_amp = True #if false, then autocast becomes no-op. No if/else in code.

start_timer()
for epoch in range(1):
    for tx, ty in zip(x,y):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(tx)
            loss = loss_fn(logits,ty)
            
        scaled_loss = grad_scaler.scale(loss)
        scaled_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=1.0
        )
        
        grad_scaler.step(opt) #unscales the grad
        grad_scaler.update()
        
        opt.zero_grad(set_to_none=True)
        
end_timer("With autograd to fp16 and gradscaler in the device {}".format(device))