import torch


a = torch.ones(3)
print(a)
a[2] = 2.0
print(a)

points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
print(points)
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)

print(points.shape)

points = torch.zeros(3, 2)
print(points)
print(points[0, 1])
