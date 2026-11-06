import torch, sys
import time
from gpt.attention import vector_attention

# Part7 benchmarking
# Use T ∈ {8, 16, 32, 64, 128, 256}, B=4, C=32, D=Dv=16, under torch.no_grad().

Ts = [8, 16, 32, 64, 128, 256]
B = 4
C = 32
D = 16

total = 4

for round in range(total):
    for T in Ts:
        x = torch.randn(size=(B,T,C))
        Wk = torch.randn(size=(D,C))
        Wq = torch.randn(size=(D,C))
        Wv = torch.randn(size=(D,C))
        start = time.perf_counter_ns()
        out, weights = vector_attention(x, Wq, Wk, Wv)
        end = time.perf_counter_ns()
        if round == total-1:
            print(f"Total time for {T:3d} is {(end-start)/1e6:5f} ms and weights {weights.nbytes/1024:.2f}KB")


"""
Quadratic increase in memory as T grows by 2, memory grows by 4
Total time for   8 is 0.132833 ms and weights 1.00KB
Total time for  16 is 0.123291 ms and weights 4.00KB
Total time for  32 is 0.115042 ms and weights 16.00KB
Total time for  64 is 0.145916 ms and weights 64.00KB
Total time for 128 is 0.388667 ms and weights 256.00KB
Total time for 256 is 0.644500 ms and weights 1024.00KB
"""
