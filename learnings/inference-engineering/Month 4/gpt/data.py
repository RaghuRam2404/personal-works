from gpt.tokenizer import CharTokenizer
import torch

def encode_and_split(
    text: str,
    tokenizer: CharTokenizer,
    train_fraction: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    encoded_data = torch.tensor(tokenizer.encode(text))
    idx = int(len(encoded_data) * train_fraction)
    train = encoded_data[:idx]
    test = encoded_data[idx : ]
    
    assert isinstance(train, torch.Tensor)
    assert isinstance(test, torch.Tensor)
    assert len(train)+len(test) == len(text)
    
    return train, test
    
    
def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    idxes = torch.randint(low=0, high=len(data)-block_size, size=(batch_size,))

    x, y = [], []
    for idx in idxes:
        idx = idx.item()
        tx = data[idx: idx+block_size]
        ty = data[idx+1: idx+block_size+1]
        x.append(tx)
        y.append(ty)
    x, y = torch.stack(x), torch.stack(y)
    
    return (x,y)