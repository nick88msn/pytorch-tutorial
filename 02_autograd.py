import torch

from torch_check import machine_check_setup

device = machine_check_setup()

# Autograd is a PyTorch package that does the gradient computation
x = torch.randn(3, requires_grad=True)
print(x)

# Now pytorch will create a computational graph to each operation so that later it is optimize to calculate the gradient during backpropagation
y = x + 2
print(y)    #now y has a grad_fn function that is used to calculate the gradient
z = y*y*2
print(z)
z = z.mean()
print(z)

# Now to calculate the gradient we just need to call the backward() function
z.backward()    # dz/dx
print(x.grad)   # x has a grad attribute where the gradients are stored

# If the tensor has a dimension greater than one, the Gradient is actually the Jacobian and that is calculated multiplying partial derivative times a vector
a = torch.randn(3, requires_grad=True)
b = a * a * 3

v = torch.tensor([1.0,0.0001,0.2], dtype=torch.float32)
b.backward(v)
print(a.grad)

# To prevent Pytorch to track the grad_fn history and waste resource computation (i.e. training) we have 3 options
# x.requires_grad_(False)
# x.detach() -> creates a new tensor without gradient
# context manager -> with torch.no_grad()

x.requires_grad_(False)
print(x)
x.requires_grad_(True)
print(x)
y = x.detach()
print(y)
x.requires_grad_(True)
with torch.no_grad():
    y = x + 2
    print(y)

# Training example
weights = torch.ones(4, requires_grad=True)
for epoch in range(2):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()