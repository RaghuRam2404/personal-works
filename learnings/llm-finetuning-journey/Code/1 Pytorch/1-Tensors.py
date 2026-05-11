import torch
import numpy as np


# direct from the array
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(type(x_data), x_data.dtype)

x_3d_data = [[[1,2],[3,4], [7,8]],[[5,6],[6,7], [8,9]]]
x_3d_data_t = torch.tensor(x_3d_data)
print(x_3d_data_t.shape)

# from numpy data
np_data = np.array(x_3d_data)
x_np_data = torch.from_numpy(np_data)
print(type(np_data), type(x_np_data), x_np_data)

# ones and zeros like
x_zeros_1 = torch.zeros_like(x_np_data)
x_zeros_2 = torch.zeros(size=x_np_data.shape)
print(x_zeros_1)
print(x_zeros_1.shape, x_zeros_2.shape)

x_ones_1 = torch.ones_like(x_np_data)
x_ones_2 = torch.ones(size=x_np_data.shape)
print(x_ones_1)
print(x_ones_1.shape, x_ones_2.shape)

# random
x_rand_1 = torch.rand(size=x_data.shape)
x_rand_2 = torch.rand_like(input=x_data, dtype=type(0.01))
print(x_rand_1, "\n", x_rand_2)
print(x_rand_1.shape)

# attributes
print("{}.shape: ",x_rand_1.shape)
print("{}.dtype: ",x_rand_1.dtype)
print("{}.device: ",x_rand_1.device)

# accelerator
print(torch.accelerator)
print(torch.accelerator.is_available())
print(torch.accelerator.current_accelerator())

# if accelerator is available move to current accelerator
x_torch = torch.rand((2,3,4))
print(x_torch, "\n", x_torch.device)
if torch.accelerator.is_available():
    x_torch = x_torch.to(torch.accelerator.current_accelerator())
    print(x_torch.device)

# indexing and slicing
print("First Row: ", x_torch[0], x_torch[0].shape)
print("First Column: ", x_torch[:,1], x_torch[:,1].shape)
print("3rd Dimension:: ", x_torch[:,:,1], x_torch[:,:,1].shape)

# concatenate
x1_torch = torch.zeros((2,3))+1
x2_torch = torch.zeros((2,3))+2
x3_torch = torch.zeros((2,3))+3
print(x1_torch, "\n", x2_torch, "\n", x3_torch)
# all tensors must be of same shape
cat1_torch = torch.cat([x1_torch, x2_torch, x3_torch], dim=0)
print(cat1_torch, cat1_torch.shape)
cat2_torch = torch.cat([x1_torch, x2_torch, x3_torch], dim=1)
print(cat2_torch, cat2_torch.shape)
# won't work
cat3_torch = torch.cat([x1_torch, x2_torch, x3_torch], dim=2)
print(cat3_torch, cat3_torch.shape)

# matrix multiplication
x1 = torch.rand((2,3))
matmul = x1 @ x1.T
print(matmul, matmul.shape)

matmul = x1.matmul(x1.T)
print(matmul, matmul.shape)

matmul = torch.matmul(x1, x1.T)
print(matmul, matmul.shape)

# element wise product
x2 = torch.randint(low=1, high=10, size=(2,3))
x1 = torch.randint(low=1, high=10, size=(2,3))
print(x1 * x2)
print(torch.mul(x1, x2))
print(x1.mul(x2))

# single element tensor
print(x1, torch.sum(x1))
print(x1.sum(), type(x1.sum()))
print(x1.sum().item(), type(x1.sum().item()))

# in place operation (ie) replace the existing variable itself
xt = torch.rand((2,3))
print(xt)
xt.t_() #transpose and storing the same
print("After inplace transpose:\n",xt)

x1 = torch.randint(low=1, high=10, size=(2,3))
print(x1, x1.add(5))
x1.add_(5)
print(x1)

# numpy and tensor share underlying memory
t = torch.randint(0,100,(10,1))
n = t.numpy()
print("t: {}, type(t): {}".format(t,type(t)))
print("n: {}, type(n): {}".format(n,type(n)))
print("Let's change 't' and see if 'n' changes")
t.add_(100) # only in-place operation does the reflection, but not the t=t+100
print("t: {}, type(t): {}".format(t,type(t)))
print("n: {}, type(n): {}".format(n,type(n)))

# create tensor from numpy
n = np.arange(0,10)
t = torch.from_numpy(n)
print("n: {}, type(n): {}".format(n,type(n)))
print("t: {}, type(t): {}".format(t,type(t)))
np.add(n, 1, out=n)
print("n: {}, type(n): {}".format(n,type(n)))
print("t: {}, type(t): {}".format(t,type(t)))