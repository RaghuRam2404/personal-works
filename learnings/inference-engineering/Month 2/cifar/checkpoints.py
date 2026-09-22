import torch
from .configs import gpu_configs
from .model import get_model

def save_checkpoint(model, optimizer, scaler, config, epochs_run, path):
    chckpt_data = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': (
            scaler.state_dict()
            if scaler is not None and scaler.is_enabled()
            else None
        ),
        'config': config,
        'epochs_run': epochs_run
    }
    
    torch.save(chckpt_data, f = path)
    
def load_checkpoint(path):

    chckpt_data = torch.load(f=path)
    model_state_dict = chckpt_data['model_state_dict']
    optimizer_dict = chckpt_data['optimizer']
    scaler_dict = chckpt_data['scaler']
    config = chckpt_data['config']
    epochs_run = chckpt_data['epochs_run']
    
    device, amp_type, use_amp, scaler = gpu_configs(config["precision_mode"])
    
    model = get_model(device)
    model.load_state_dict(model_state_dict)
    
    #optimizer holds reference to the model's parameter, so it doesn't have .to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    optimizer.load_state_dict(optimizer_dict)
    
    if scaler is not None and use_amp:
        scaler.load_state_dict(scaler_dict)
    
    return model, optimizer, scaler, config, device, amp_type, use_amp, epochs_run
    

def verify_checkpoint(model, path, data, device, amp_type, use_amp):
    m2, optimizer, scaler, config, device, amp_type, use_amp, epochs_run = load_checkpoint(path=path)
    with torch.no_grad():
        model = model.to(device)
        m2 = m2.to(device)
        
        model.eval()
        m2.eval()
        
        for x,y in data:
            x = x.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_type, enabled=use_amp):
                logits1 = model(x)
                logits2 = m2(x)
            
            print(torch.allclose(logits1, logits2))
            print((logits1-logits2).abs().max())
            assert torch.equal(logits1, logits2), (
                f"Checkpoint mismatch; max absolute difference: {(logits1-logits2).abs().max()}"
            )
            break
       