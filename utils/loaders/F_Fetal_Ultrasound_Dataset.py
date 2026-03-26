from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Fetal_Ultrasound_Dataset(Dataset):
	def __init__(self, csv_file, images_dir, transform=None, filter_class=None):
		self.data = pd.read_csv(csv_file)
		self.images_dir = Path(images_dir)
		self.transform = transform

		if filter_class:
			self.data = self.data[self.data["label"] == filter_class]
			self.data = self.data.reset_index(drop=True)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		row = self.data.iloc[idx]
		img_name = row["id_image"]
		img_path = self.images_dir / img_name

		image = Image.open(img_path).convert("L")

		if self.transform:
			image = self.transform(image)

		return image, image, row["label"]
