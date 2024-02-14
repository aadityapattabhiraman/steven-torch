import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# hello hello can you hear me


class Neural_Network(nn.Module):
	def __init__(self, input_size, num_classes):
		super(Neural_Network, self).__init__()
		self.layer_1 = nn.Linear(input_size, 50)
		self.layer_2 = nn.Linear(50, num_classes)

	def forward(self, x):
		x = functional.relu(self.layer_1(x))
		x = self.layer_2(x)
		return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 784
num_classes = 10
learning_rate = 0.001
num_epochs = 3 

batch_size = 64
train_dataset = datasets.MNIST( 
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True,
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(),
    download=True,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
)

model = Neural_Network(input_size=input_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
	print(f"Epoch: {epoch}")

	for batch_idx, (data, targets) in enumerate(train_loader):
		data = data.to(device=device)
		targets = targets.to(device=device)

		data = data.reshape(data.shape[0], -1)

		scores = model(data)
		loss = criterion(scores, targets)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()


def check_accuracy(loader, model):
	num_correct = 0 
	num_samples = 0
	model.eval()

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device=device)
			y = y.to(device=device)
			x = x.reshape(x,shape[0], -1)

			scores = model(x)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)

		print(
			f"Got {num_correct}/{num_samples} with accuracy"
			f" {float(num_correct)}/float(num_samples) * 100:.2f")

	model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)