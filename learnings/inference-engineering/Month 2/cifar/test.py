import torch

def get_accuracy(model, dataloader, device, amp_type, use_amp):
    model.to(device)
    with torch.no_grad():
        model.eval()
        total_correct = 0
        for idx, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_type, enabled=use_amp):
                logits = model(x)
            y_hat = logits.argmax(dim=1)

            total_correct += torch.sum((y==y_hat).float()).item()
        accuracy = total_correct / len(dataloader.dataset)
        
    return accuracy
 