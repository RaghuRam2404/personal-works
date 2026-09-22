import torch, random, platform
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

from cifar.env import env_check, set_seed
from cifar.configs import gpu_configs
from cifar.dataloaders import build_dataloaders, print_sample_data
from cifar.model import get_model
from cifar.training import train_one_epoch, train_loop
from cifar.checkpoints import save_checkpoint, load_checkpoint, verify_checkpoint
from cifar.test import get_accuracy

MODE = 'NEW' # NEW or RESUME
#MODE = 'RESUME'

if MODE == "NEW":

    config = {
        "seed": 42, 
        "batch_size": 128, 
        "epochs": 5, 
        "learning_rate": 0.01, 
        "precision_mode": "fp32", #fp32, bf16, fp16
        "workers": 0
    }


    device, amp_type, use_amp, scaler = gpu_configs(config["precision_mode"])
    env_check()
    set_seed(config["seed"])

    train_loader, test_loader, classes = build_dataloaders(config["batch_size"], config["workers"])
    #print_sample_data(dataloader=train_loader, classes=classes)
    model = get_model(device=device)
    optimizer, scaler, losses = train_loop(model=model, learning_rate=config["learning_rate"], 
                        train_loader=train_loader, 
                        epochs=config["epochs"],
                        optimizer=None,
                        device=device, use_amp=use_amp, scaler=scaler, amp_type=amp_type, epochs_run=0)

    file_name = "./cifar10_ck1.pth"
    save_checkpoint(model, optimizer, scaler, config, config['epochs'], file_name)
    accuracy = get_accuracy(model, test_loader, device, amp_type, use_amp)
    print(f"Accuracy: {accuracy:.4f}")
    verify_checkpoint(model, file_name, test_loader, device=device, amp_type=amp_type, use_amp=use_amp)

else:
    # This is to resume from a checkpoint
    chckpt_file = "./cifar10_ck1.pth"
    
    model, optimizer, scaler, config, device, amp_type, use_amp, epochs_run = load_checkpoint(chckpt_file)
    train_loader, test_loader, classes = build_dataloaders(config["batch_size"], config["workers"])
    optimizer, scaler, losses = train_loop(model=model, learning_rate=config["learning_rate"], 
                        train_loader=train_loader, 
                        epochs=config["epochs"],
                        optimizer = optimizer,
                        device=device, use_amp=use_amp, scaler=scaler, amp_type=amp_type, epochs_run=epochs_run)
    save_checkpoint(model, optimizer, scaler, config, epochs_run+config['epochs'], chckpt_file)
    
    accuracy = get_accuracy(model, test_loader, device, amp_type, use_amp)
    print(f"Accuracy: {accuracy:.4f}")
