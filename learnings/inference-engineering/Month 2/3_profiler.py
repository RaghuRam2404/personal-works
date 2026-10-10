import torch, time
from torchvision.models import resnet18
from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler import schedule

torch.cuda.is_available()

model = resnet18()
parameters = [a for a, b in model.named_parameters()]
len(parameters)

random_input = torch.randn(size=(5,3,224, 224)) #3 is the channel count

def get_sort_by_keys(event):
    sort_by_keys = [name for name in dir(event) if not name.startswith("__") and any (term in name for term in ('time', 'memory', 'count')) ]
    return sort_by_keys

###########################################

start_time = time.perf_counter_ns()
logits = model(random_input)
end_time = time.perf_counter_ns()
total_time = (end_time-start_time)/(1000*1000)
print(f"Total time taken checked naively: {total_time} milli seconds")

###########################################

### Monitoring thhe time
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities += [ProfilerActivity.CUDA]
with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
    with record_function("inference"):
        logits = model(random_input)

prof.export_chrome_trace("trace.json")

### Analysis

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
help(prof.key_averages())

event = prof.key_averages()[0]
sort_by_keys = get_sort_by_keys(event)
print(sort_by_keys)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))

print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=20))

###########################################

### Monitoring thhe time & memory in GPU
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities += [ProfilerActivity.CUDA]

with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
    model = resnet18().to('cuda')
    random_input = random_input.to('cuda')
    with record_function("inference"):
        logits = model(random_input)
prof.export_chrome_trace("cuda_trace.json")

from pathlib import Path
trace_path = Path("cuda_trace.json").resolve()

from google.colab import files
files.download("/content/cuda_trace.json")

###########################################

## FOR LONG RUNNING JOBS


## For this one, weh ave to set numbers after some trial to see how much cycles are taking for initial
## kernel load, data transfer etc
## wait & warmup will come inside the cycle
## skip happens based on prof.step() but not based on the internal operations
## With the below config and 10 steps, we'll get output for 4, 7, 10
scheduler = schedule(skip_first=1, wait=1, warmup=1, active=1, repeat=4)

def trace_handler(prof):
    step = prof.step_num
    data = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    print(data)
    prof.export_chrome_trace("./tmp/trace_{}.json".format(step))
    
with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True,
             schedule=scheduler, on_trace_ready=trace_handler) as prof:
    
    for epoch in range(10):
        logits = model(random_input)
        prof.step()
    
    
###########################################

## Instrumentation


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
num_batches = 1
num_layers = 3
epochs = 5

with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
    
    with record_function("pre_training_work"):
        x,y = build_data(batch_size, in_size, out_size, num_batches)
        model = build_model(in_size, out_size, num_layers)
        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.SGD(params=model.parameters(), lr=0.001)

    with record_function("training_loop"):
        for epoch in range(2): #each epoch
            
            with record_function("one_training_loop"):
                for tx,ty in zip(x,y): #each batch
                    
                    with record_function("forward_pass"):
                        logits = model(tx)
                        loss = loss_fn(logits, ty)
                    
                    with record_function("backward_pass"):
                        loss.backward()
                    
                    with record_function("update_grad"):
                        opt.step()
                        
                    with record_function("zero_grad"):
                        opt.zero_grad(set_to_none=True)
                        
prof.export_chrome_trace("training_loop.json")
