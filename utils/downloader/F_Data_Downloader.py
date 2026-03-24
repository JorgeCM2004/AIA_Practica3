import csv
import shutil
from pathlib import Path

import kagglehub


class Data_Downloader:
	def download(self, force_download=False):
		data_dir = Path(__file__).parent.parent.parent / "data"
		images_dir = data_dir / "images"
		csv_path = data_dir / "fetal_ultrasound.csv"

		if not force_download and csv_path.exists() and images_dir.exists():
			return

		nested_dir = (
			data_dir
			/ "Ultrasound Fetus Dataset"
			/ "Ultrasound Fetus Dataset"
			/ "Data"
			/ "Data"
		)

		shutil.rmtree(data_dir, True)
		data_dir.mkdir(exist_ok=True)

		kagglehub.dataset_download(
			"orvile/ultrasound-fetus-dataset",
			force_download=force_download,
			output_dir=str(data_dir),
		)

		images_dir.mkdir(exist_ok=True)

		with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["id_image", "label"])

			for split in ["train", "test", "validation"]:
				split_dir = nested_dir / split
				if split_dir.exists():
					for label_dir in split_dir.iterdir():
						if label_dir.is_dir():
							for img_path in label_dir.iterdir():
								if img_path.is_file():
									shutil.move(
										str(img_path), str(images_dir / img_path.name)
									)
									writer.writerow([img_path.name, label_dir.name])

		shutil.rmtree(data_dir / "Ultrasound Fetus Dataset", ignore_errors=True)
		shutil.rmtree(data_dir / ".complete", ignore_errors=True)
		(data_dir / "ultrasound_fetus.csv").unlink(missing_ok=True)
