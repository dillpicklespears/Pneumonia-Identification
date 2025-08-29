# following along: 
# https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

# PyTorch has a built-in differentiation engine called torch.autograd. 
# It supports automatic computation of gradient for any computational graph.

import torch

# Consider the simplest one-layer neural network, with input x, 
# parameters w and b, and some loss function. 
# It can be defined in PyTorch in the following manner:

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# In this network, w and b are parameters, which we need to optimize. 
# Thus, we need to be able to compute the gradients of loss function 
# with respect to those variables. 
# In order to do that, we set the requires_grad property of those tensors.

# A function that we apply to tensors to construct computational graph 
# is in fact an object of class Function. 
# This object knows how to compute the function in the forward direction, 
# and also how to compute its derivative during the backward propagation step.
# A reference to the backward propagation function is stored in grad_fn property of a tensor.

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


# COMPUTING GRADIENTS 
# --------------------- 

# To optimize weights of parameters in the neural network, 
# we need to compute the derivatives of our loss function with respect to parameters
# To compute those derivatives, we call loss.backward(), 
# and then retrieve the values from w.grad and b.grad:

loss.backward()
print(w.grad)
print(b.grad)


# DISABLING GRADIENT TRACKING
# --------------------- 

# all tensors with requires_grad=True support gradient computation. 
# However, there are some cases when we do not need to do that, 
# for example, when we have trained the model and just want to apply it to some input data, 
# i.e. we only want to do forward computations through the network. 
# We can stop tracking computations by surrounding our computation code with torch.no_grad() block:

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Another way to achieve the same result is to use the detach() method on the tensor:

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# There are reasons you might want to disable gradient tracking:
# * To mark some parameters in your neural network as frozen parameters.
# * To speed up computations when you are only doing forward pass, 
#   because computations on tensors that do not track gradients would be more efficient.



