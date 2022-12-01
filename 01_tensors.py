import torch

# local imports
from torch_check import machine_check_setup

device = machine_check_setup()

# Create a tensor directly on the MPS device
# torch.empty, torch.zeros, torch.ones, torch.rand
x = torch.rand(5,5, dtype=torch.float16, device=device)
# to find values type
print(f"x is made of {x.dtype}")    # types = int, float, float16, double, float32
# to check size
print(x.size())

# Create a tensor from a variable
y = torch.tensor([
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1]
], device=device)

# Basic operations with tensors
a = torch.rand(2,2)
b = torch.rand(2,2)

c = a + b   # or torch.add(x,y)
print(c)

# To modify b adding a
b.add_(a)   #In torch every function with _ will do a in-trailing operation
print(b)

# Same with substraction
d = a - b
d = torch.sub(a,b)
print(d)

a.sub_(b)
print(a)

# Multiplication
e = a * b
e = torch.mul(a,b)
print(e)
b.mul_(a)
print(b)

# Division
f = a/b
f = torch.div(a,b)
print(f)

# Slicing
u = torch.rand(5,3)
print(x)
s = x[:,0]
print(s)
print(s.size())
t = x[1,:]
print(t, t.size())
h = x[1,1]
print(h)
print(h.size()) #Still a tensor to get the actual value use .item() method
print(h.item())

# Reshaping a Tensor
aa = torch.rand(3,2)
print(a)
bb = aa.view(6) # it has to contain the same elements number of aa
print(bb, bb.size())

# to automatically resize one shape use -1
cc = aa.view(-1)
print(cc, cc.size())

# Converting from Numpy array to torch Tensors and vice versa
import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()   #converting using torch numpy method
print(b, type(b))

# if a and b are on the same device (cpu, gpu) then they share the same memory space, so changing one (a) will change the other (b)
a.add_(1)
print(a)
print(b)

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

# Modifying the np array now we change also the torch tensor, if they are in the same device (cpu, gpu)
a+=1
print(b)

# Numpy only handles CPU array, To change torch device
cc = torch.ones(5, device="cpu")
dd = cc.numpy()
cc = cc.to("mps")
print(cc)
print(dd)
# Now we can change the torch Tensor without modifying the np array
cc.add_(2)
print(cc)
print(dd)
