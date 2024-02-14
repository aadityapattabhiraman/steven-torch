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

# simple comparison
z = x < 0
z = x > 0

# matmul
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)

# matrix exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)

# element wise multiplication
z = x * y 
print(z)