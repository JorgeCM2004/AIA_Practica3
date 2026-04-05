import torch.nn as nn


class CNN_Classifier(nn.Module):
	def __init__(self, in_channels=1, num_classes=2):
		super().__init__()

		self.network = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			nn.Linear(64 * 28 * 28, 128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128, num_classes),
		)

	def forward(self, x):
		x = self.network(x)
		return x
