import torch


x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.empty(3)
z = x + y 
z = x - y 
z = torch.true_divide(x, y)

# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x

# exponential
z = x.pow(2)
z = x ** 2