import torch

class CharTokenizer:
    
    def __init__(self, dictionary:str, handle_unknown:bool = False):
        chars = set()
        for c in dictionary:
            chars.add(c)
        self.handle_unknown = handle_unknown
        if self.handle_unknown:
            self.UNKNOWN = "<UNK>"
            chars.add(self.UNKNOWN)
        
        #print(sorted(chars))
        self.chars = sorted(chars)
        self.itos = {}
        self.stoi = {}
        for idx, c in enumerate(self.chars):
            self.itos[idx] = c
            self.stoi[c] = idx
    
    def encode(self, text) -> list[int]:
        encoded = []
        for c in text:
            if c not in self.chars:
                if self.handle_unknown:
                    c = self.UNKNOWN
                else:
                    raise ValueError("Got unknown character during encoding")
            encoded.append(self.stoi[c])
        return encoded
    
    def decode(self, ids) -> str | torch.Tensor:
        text = []
        nchars = self.vocab_size
        for id in ids:
            if id >= 0 and id < nchars:
                text.append(self.itos[id])
            else:
                if self.handle_unknown:
                    text.append(self.UNKNOWN)
                else:
                    raise ValueError("Got unknown character during encoding")
        return "".join(text)
    
    @property
    def vocab_size(self) -> int:
        if self.chars is not None:
            return len(self.chars)
        raise ValueError("Vocabulary not initialized yet")