from gpt.data import get_batch
import torch
import torch.nn as nn

class BigramLanguageModel(torch.nn.Module):
    
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
    
    def forward(self, idx):
        return self.embedding(idx)


    @torch.no_grad()
    def generate(self, idx_input, max_new_tokens):
        print(idx_input.shape)
        for _ in range(max_new_tokens):
            last_row_logits = self(idx_input[:,-1])
            probs = nn.Softmax(dim=1)(last_row_logits)
            nidx = torch.multinomial(probs, num_samples=1, replacement=True)
            idx_input = torch.cat((idx_input, nidx), dim=1)
        return idx_input


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, eval_batches, loss_fn):
    model.eval()
    losses = torch.zeros(eval_batches)
    for k in range(eval_batches):
        ix, iy = get_batch(data, batch_size, block_size)
        logits = model(ix)
        losses[k] = loss_fn(logits.view(-1, logits.shape[-1]), iy.view(-1))
    model.train()
    
    return losses.mean().item()
    
### TESTER CODES

torch.manual_seed(10)
vocab_size = 10
model = BigramLanguageModel(vocab_size)
model.train()
batch, tokens = 2, 3
logits = model(torch.randint(low=0, high=vocab_size, size=(batch, tokens)))
print(f"{logits.shape=}")

batch, prev = (2,3)
idx_input = torch.randint(low=0, high=vocab_size, size=(batch, prev))
idx_input[:,-1]
model.generate(idx_input, max_new_tokens=5)
    

emb = nn.Embedding(3,3)
emb(torch.tensor([[0,0,1],[0,1,2]]))

torch.manual_seed(10)
array = torch.randn((2,3,4))
array[:,-1,:].shape

nn.Softmax(1)(torch.randn((1,10)))