import torch, random, platform
import numpy as np

def env_check():
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
        print("BF16 native:", torch.cuda.is_bf16_supported(including_emulation=False))
        print("cuDNN:", torch.backends.cudnn.version())


def set_seed(seed):
    torch.manual_seed(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True