import torch

def gpu_configs(precision_mode):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    amp_type = torch.float32
    use_amp = False
    scaler = None

    if precision_mode == "fp16":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            amp_type = torch.float16
            use_amp = True
            scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
        elif torch.accelerator.is_available():
            device = torch.accelerator.current_accelerator()
            amp_type = torch.float16
            use_amp = True
        else:
            raise RuntimeError("No option available to use fp16 for autocast")
        
    elif precision_mode == "bf16":
        if not torch.cuda.is_available():
            raise RuntimeError("BF16 AMP requires a CUDA device for this experiment.")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(
                "Native CUDA BF16 is not supported on this GPU. "
                "Use fp32 or fp16 and record the hardware limitation."
            )
        
        device = torch.device("cuda")
        amp_type = torch.bfloat16
        use_amp = True

    return device, amp_type, use_amp, scaler