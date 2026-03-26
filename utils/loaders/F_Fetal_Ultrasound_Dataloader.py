from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from .F_Fetal_Ultrasound_Dataset import Fetal_Ultrasound_Dataset


class Fetal_Ultrasound_Dataloader:
	def __init__(self, data_dir="data", batch_size=32, img_size=(224, 224)):
		self.data_dir = Path(data_dir)
		self.images_dir = self.data_dir / "images"

		self.train_csv = self.data_dir / "fetal_ultrasound_train.csv"
		self.test_csv = self.data_dir / "fetal_ultrasound_test.csv"

		self.batch_size = batch_size
		self.img_size = img_size

	def get_transforms(self):
		return transforms.Compose(
			[
				transforms.Resize(self.img_size),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5], std=[0.5]),
			]
		)

	def create_dataloaders(self):
		transform = self.get_transforms()

		train_dataset = Fetal_Ultrasound_Dataset(
			csv_file=self.train_csv,
			images_dir=self.images_dir,
			transform=transform,
			filter_class="benign",
		)

		test_dataset = Fetal_Ultrasound_Dataset(
			csv_file=self.test_csv,
			images_dir=self.images_dir,
			transform=transform,
			filter_class=None,
		)

		train_loader = DataLoader(
			train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=4,
			pin_memory=True,
		)

		test_loader = DataLoader(
			test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=4,
			pin_memory=True,
		)

		return train_loader, test_loader
