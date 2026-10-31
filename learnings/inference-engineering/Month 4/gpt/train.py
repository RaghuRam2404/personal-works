from gpt.bigram import BigramLanguageModel
from gpt.data import encode_and_split, get_batch
from gpt.tokenizer import CharTokenizer
from pathlib import Path

with open(Path("shakespeare.txt")) as f:
    text = f.read()

ct = CharTokenizer(dictionary=text)
train, test = encode_and_split(text, ct)
model = BigramLanguageModel(vocab_size=ct.vocab_size)
print(get_batch(train, 4, 3))