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


# squeeze abd unsqueeze
points_64 = torch.rand(5, dtype=torch.double)
print(points_64)

a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
print(a.shape, a_t.shape)

# .storage()

# .zero_()

# .storage_offset(), .stride()

# .clone() subtensor into a new tensor

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)
points_t = points.t()
print(points_t)
# transpose of a tensor

# contiguous tensor
# A tensor whose values are laid out in the storage starting from the rightmost
# dimension onward
print(points.is_contiguous())

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
points_t_cont = points_t.contiguous()

points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
points_gpu = points.to(device='cuda')