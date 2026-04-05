import torch
import torch.nn as nn
from torchvision import models


class ResNet_Classifier(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()

		self.network = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

		for param in self.network.parameters():
			param.requires_grad = False

		original_weights = self.network.conv1.weight.clone()
		self.network.conv1 = nn.Conv2d(
			1, 64, kernel_size=7, stride=2, padding=3, bias=False
		)
		with torch.no_grad():
			self.network.conv1.weight.copy_(
				torch.sum(original_weights, dim=1, keepdim=True)
			)

		num_ftrs = self.network.fc.in_features
		self.network.fc = nn.Sequential(
			nn.Dropout(p=0.5), nn.Linear(num_ftrs, num_classes)
		)

	def forward(self, x):
		return self.network(x)
