# following along: 
# https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

import torch
import numpy as np

# CREATING A TENSOR 
# ---------------------

# directly from data: 
data = [[1, 2],[3, 4]]  
x_data = torch.tensor(data) # data type is inferred 

# from numpy arrays:
np_array = np.array(data) 
x_np = torch.from_numpy(np_array)

# from another tensor:
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# with random or constant values:
shape = (2,3,) # tuple of tensor dimensions, determines the dimensionality of the output tensor
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# ATTRIBUTES OF A TENSOR  
# ---------------------

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# OPERATIONS ON TENSORS
# ---------------------

# tensor operations can be run on the CPU and the accelerator 

# By default, tensors are created on the CPU
# We need to explicitly move tensors to the accelerator using .to method 
# copying large tensors across devices can be expensive

# accelerators are devices that speed up operations 
# usually by running them in parrelel 
# we can assume that a device only has one accelerator 

# We move our tensor to the current accelerator if available:
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

# standard numpy-like indexing and slicing:
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# torch.cat to concatenate a sequence of tensors along a given dimension:
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# arithmetic operations:
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# single element tensors can be made by aggregating all values of a tensor into one value: 
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


# BRIDGE WITH NUMPY
# ---------------------

# Tensors on the CPU and NumPy arrays can share their underlying memory locations

# tensor to numpy arrawy: 
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array:
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# numpy array to tensor: 
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")