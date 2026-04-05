from typing import Literal, get_args

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .F_CNN_Classifier import CNN_Classifier
from .F_ResNet_Classifier import ResNet_Classifier

POSSIBLE_MODELS = Literal["CNN_Classifier", "ResNet_Classifier"]


class Classifier_Manager:
	def __init__(
		self,
		models: list[get_args(POSSIBLE_MODELS)] | Literal["All"] = "All",
		seed=None,
	):
		if models == "All":
			models = list(get_args(POSSIBLE_MODELS))

		self.models_names = [m for m in models if m in set(get_args(POSSIBLE_MODELS))]
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.models = {}

		self.encoder = LabelEncoder()
		self.encoder.fit(["benign", "malignant"])

		for name in self.models_names:
			match name:
				case "CNN_Classifier":
					self.models[name] = CNN_Classifier().to(self.device)
				case "ResNet_Classifier":
					self.models[name] = ResNet_Classifier().to(self.device)

	def train(self, data_loader, epochs=20, lr=1e-3):
		for name, model in self.models.items():
			if name in ["CNN_Classifier", "ResNet_Classifier"]:
				self._train_classificator(model, data_loader, epochs, lr)

	def _train_classificator(self, model, loader, epochs, lr):
		weights = torch.tensor([2.0, 1.0]).to(self.device)
		criterion = nn.CrossEntropyLoss(weight=weights)
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

		for epoch in range(epochs):
			model.train()
			loop = tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=True)

			correct = 0
			total = 0

			for inputs, _, labels in loop:
				inputs = inputs.to(self.device)

				if isinstance(labels[0], str):
					num_labels = self.encoder.transform(
						[label.lower() for label in labels]
					)
				else:
					num_labels = [int(label) for label in labels]

				targets = torch.tensor(num_labels, dtype=torch.long).to(self.device)

				optimizer.zero_grad()
				outputs = model(inputs)

				loss = criterion(outputs, targets)
				loss.backward()
				optimizer.step()

				_, predicted = torch.max(outputs.data, 1)
				total += targets.size(0)
				correct += (predicted == targets).sum().item()

				acc = 100 * correct / total
				loop.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

	def predict(self, loader):
		out = {}
		for model_name in self.models_names:
			model = self.models[model_name]
			model.eval()
			y_pred = []
			y_true = []

			with torch.no_grad():
				for inputs, _, labels in loader:
					inputs = inputs.to(self.device)
					outputs = model(inputs)

					_, predicted = torch.max(outputs.data, 1)
					y_pred.extend(predicted.cpu().numpy())

					if isinstance(labels[0], str):
						y_true.extend(
							self.encoder.transform([label.lower() for label in labels])
						)
					else:
						y_true.extend([int(label) for label in labels])
			out[model_name] = (y_true, y_pred)
		return out
