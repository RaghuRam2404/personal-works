import torch
import time

def train_one_epoch(model, train_loader, optimizer, loss_function, scaler, device, amp_type, use_amp):
    temp_losses = []
    for idx, (x, y) in enumerate(train_loader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device.type, dtype=amp_type, enabled=use_amp):
            logits = model(x)
            loss = loss_function(logits, y)
        
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        temp_losses.append(loss.item())
    return temp_losses
    

def train_loop(model, train_loader, epochs, learning_rate, optimizer, device, use_amp, amp_type, scaler, epochs_run):
    if optimizer is None:
        optimizer = torch.optim.SGD(params = model.parameters(), lr=learning_rate)
        
    loss_function = torch.nn.CrossEntropyLoss()

    model.train()
    losses = []

    for epoch in range(epochs):
        start = time.time()
        temp_losses = train_one_epoch(model, train_loader, optimizer, loss_function, scaler, device, amp_type, use_amp)
        end = time.time()
        time_taken = (end-start)*1000
        avg_loss = sum(temp_losses)/len(temp_losses)
        print(f"Epoch [{(epochs_run+epoch+1):3d}/{epochs_run+epochs:3d}] : Loss {avg_loss:.5f} in {time_taken}ms")
        losses.append(avg_loss)
        
    return optimizer, scaler, losses
