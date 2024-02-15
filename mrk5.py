import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# hello hello can you hear m
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2


class Recurrent_Neural_Network(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(Recurrent_Neural_Network, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

		out, _ = self.rnn(x, h0)
		out = out.reshape(out.shape[0], -1)
		out = self.fc(out)
		return out


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

model = Recurrent_Neural_Network(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
	print(f"Epoch: {epoch}")

	for batch_idx, (data, targets) in enumerate(train_loader):
		data = data.to(device=device).squeeze(1)
		targets = targets.to(device=device)

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
			x = x.to(device=device).squeeze(1)
			y = y.to(device=device)

			scores = model(x)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)

		print(
			f"Got {num_correct}/{num_samples} with accuracy"
			f" {float(num_correct)/float(num_samples) * 100:.2f}")

	model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)