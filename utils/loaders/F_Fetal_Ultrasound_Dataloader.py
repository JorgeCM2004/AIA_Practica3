from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional

from .F_Fetal_Ultrasound_Dataset import Fetal_Ultrasound_Dataset


class Fetal_Ultrasound_Dataloader:
	def __init__(self, data_dir="data", batch_size=32, img_size=(224, 224)):
		self.data_dir = Path(data_dir)
		self.images_dir = self.data_dir / "images"

		self.train_csv = self.data_dir / "fetal_ultrasound_train.csv"
		self.test_csv = self.data_dir / "fetal_ultrasound_test.csv"

		self.batch_size = batch_size
		self.img_size = img_size
		self.train_transform = transforms.Compose(
			[
				transforms.Lambda(self._fill_image),
				transforms.Resize(self.img_size),
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.RandomRotation(degrees=15),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5], std=[0.5]),
			]
		)
		self.test_transform = transforms.Compose(
			[
				transforms.Lambda(self._fill_image),
				transforms.Resize(self.img_size),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5], std=[0.5]),
			]
		)

	def _fill_image(self, image):
		w, h = image.size
		max_wh = max(w, h)
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return functional.pad(image, padding, 0, "constant")

	def create_dataloaders(self):
		train_dataset = Fetal_Ultrasound_Dataset(
			csv_file=self.train_csv,
			images_dir=self.images_dir,
			transform=self.train_transform,
			filter_class=None,
		)

		test_dataset = Fetal_Ultrasound_Dataset(
			csv_file=self.test_csv,
			images_dir=self.images_dir,
			transform=self.test_transform,
			filter_class=None,
		)

		train_loader = DataLoader(
			train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=4,
			pin_memory=True,
			persistent_workers=True,
		)

		test_loader = DataLoader(
			test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=4,
			pin_memory=True,
			persistent_workers=True,
		)

		return train_loader, test_loader
