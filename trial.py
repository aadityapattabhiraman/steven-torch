from Neural_Network.Neural_Network import train, neural_network
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

model = neural_network()
# x = torch.randn(64,784)
# print(model(x).shape)
train_dataset = datasets.MNIST( 
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True,
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=32, shuffle=True
)
train(train_loader, model)