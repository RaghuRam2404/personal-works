import torch
import os
from gpt.tokenizer import CharTokenizer
from gpt.data import encode_and_split, get_batch
from gpt.bigram import BigramLanguageModel, estimate_loss

with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

### TESTER CODES

len(text)
repr(text[:200])
len(set(text))
    
ct = CharTokenizer(text)
print(ct.vocab_size)
print(ct.encode("Shakes"))
print(ct.decode([31, 46, 39, 49, 43, 57]))

train, test = encode_and_split("TestStringgaddads", ct)
print(train, test)
get_batch(torch.tensor([1,2,3,4,5,6,7,8,9,10]), batch_size=10, block_size=4)

model = BigramLanguageModel(vocab_size=ct.vocab_size)

ct = CharTokenizer(dictionary=text)
train, test = encode_and_split(text, ct)
model = BigramLanguageModel(vocab_size=ct.vocab_size)
ix, iy = get_batch(train, 4, 3)
loss_fn = torch.nn.CrossEntropyLoss()

logits = model(ix)
loss = loss_fn(logits.view(-1, logits.shape[-1]), iy.view(-1))

### SHAKESPEARE RUN

seed = 1337
batch_size = 32
block_size = 8
learning_rate = 1e-2
max_steps = 2000
eval_interval = 200
eval_batches = 20
losses = []

torch.manual_seed(seed)
ct = CharTokenizer(dictionary=text)

train, test = encode_and_split(text, ct)
model = BigramLanguageModel(vocab_size=ct.vocab_size)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

input_idx = torch.zeros(size=(1,1), dtype=torch.int64)
pred = model.generate(input_idx, max_new_tokens=300)[0]
words_pred = ct.decode(pred.tolist())
### \nSKIcLT;AcELMoTbvZv C?nq-QE33:CJqkOKH-q;:la!oiywkHjgChzbQ?u!3bLIgwevmyFJGUGp\nwnYWmnxKWWev-tDqXErVKLgJt-wBpm&yiltNCjeO3:Cx&vvMYW-txjuAd IRFbTpJ$zkZelxZtTlHNzdXXUiQQY:qFINTOBNLI,&oTigq z.c:Cq,SDXzetn3XVjX-YBcHAUhk&PHdhcOb\nnhJ?FJU?pRiOLQeUN!BxjPLiq-GJdUV'hsnla!murI!IM?SPNPq?VgC'R\npD3cLv-bxn-tL!upg\nSZ!Uv

for epoch in range(max_steps):
    opt.zero_grad()
    
    ix, iy = get_batch(train, batch_size=batch_size, block_size=block_size)
    logits = model(ix)
    loss = loss_fn(logits.view(-1, logits.shape[-1]), iy.view(-1))
    losses.append(loss.item())
    
    loss.backward()
    opt.step()
    
    if (epoch+1) % eval_interval == 0:
        with torch.no_grad():
            model.eval()
            tr = estimate_loss(model, train, batch_size, block_size, eval_batches, loss_fn)
            va = estimate_loss(model, test, batch_size, block_size, eval_batches, loss_fn)
            
            #ix, iy = get_batch(train, eval_batches, block_size)
            #train_loss = loss_fn(model(ix).view(-1, ct.vocab_size), iy.view(-1))
            #ix, iy = get_batch(test, eval_batches, block_size)
            #test_loss = loss_fn(model(ix).view(-1, ct.vocab_size), iy.view(-1))
            print(f"Epoch {epoch+1:5d}/{max_steps:5d} -> train_loss: {tr:.5f}, test_loss: {va:.5f}")
            model.train()
            
#input_idx = torch.randint(low=0, high=ct.vocab_size-1, size=(5,1), dtype=torch.int64)
input_idx = torch.zeros(size=(1,1), dtype=torch.int64)
pred = model.generate(input_idx, max_new_tokens=300)[0]
words_pred = ct.decode(pred.tolist())
print(words_pred)
"""

Thand, sowht ClanBy-r har
KINRD 'swind
We, Palion, s w pro y ben dis ounerrofrssple tim; stotwoour menerad,
OMOfofo atas ke jen t!
y bor he n hesinomarstitha heap'de.
MENo min ius m:
We Spad orim; fomecu tirors, s h ENGee, h, wOF tiRDe s por, ous, achirery wikn torsw ts, ar me, theed IORofonn

HEY C
"""

with torch.no_grad():
    model.eval()
    train_loss = loss_fn(model(train[:-1]), train[1:])
    test_loss = loss_fn(model(test[:-1]), test[1:])
print(f"train_loss: {train_loss}, test_loss: {test_loss}")
# train_loss: 2.463697671890259, test_loss: 2.4915611743927