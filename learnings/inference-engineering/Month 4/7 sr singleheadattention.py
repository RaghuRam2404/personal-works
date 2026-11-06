# https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
# only the single head attention part

import torch
import torch.nn as nn

sentence = "Life is short, eat dessert first"

words = sorted(sentence.replace(",","").split(" "))
stoi = {word:idx for idx, word in enumerate(words)}

#encode
encoded = torch.tensor([stoi[word] for word in sentence.replace(",","").split(" ")])

# %%
embed_size = 16
torch.manual_seed(123)
embed_layer = nn.Embedding(len(encoded), embed_size)
embed = embed_layer(encoded).detach()
print(embed)
print(embed.shape)

# %%
torch.manual_seed(123)
d = embed_size
dk, dq, dv = 24, 24, 28

# query, key, value parameters
W_query = torch.rand((dq, d), requires_grad=False)
W_key = torch.rand((dk, d), requires_grad=False)
W_value = torch.rand((dv, d), requires_grad=False)

print(f"{W_query.shape=}, variance:{W_query.var()},\n{W_key.shape=}, variance:{W_key.var()},\n{W_value.shape=}, variance:{W_value.var()}\n")
embed.shape

# finding q,k,v for a single token at time T
index = 2
query_2 = W_query @ embed[index]
key_2 = W_key @ embed[index]
value_2 = W_value @ embed[index]

print(f"{query_2.shape=},\n{key_2.shape=},\n{value_2.shape=}\n")

# finding q,k,v for the input sequence
query = embed @ W_query.T
key = embed @ W_key.T
value = embed @ W_value.T
print(f"{query.shape=},{query.var()=},\n{key.shape=},{key.var()=},\n{value.shape=},{value.var()=}\n")

omega_24 = torch.dot(query[1], key[4])
print(omega_24) # scalar value

omega = query @ key.T #it is unnormalized
print("\n======= Unnormalized attention weights")
print(f"{omega[1]=}", "\n", omega.shape)
print(f"{omega.var()=}")

scaled_omega = omega/dk**0.5
print(f"{scaled_omega.var()=}")
attention_weights = torch.softmax(scaled_omega, dim=1)
print("\n======= Normalized Attention weights")
print(f"{attention_weights[1]=}","\n", attention_weights.shape)

print("\n======= Context Vector")
context_vector = (attention_weights @ value)
print(f"{context_vector[1]=}", "\n", context_vector.shape)

# %%