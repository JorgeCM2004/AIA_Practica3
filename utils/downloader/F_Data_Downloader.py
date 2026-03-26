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
			/ "Datasets"
		)

		shutil.rmtree(data_dir, ignore_errors=True)
		data_dir.mkdir(exist_ok=True)

		kagglehub.dataset_download(
			"orvile/ultrasound-fetus-dataset",
			force_download=force_download,
			output_dir=str(data_dir),
		)

		images_dir.mkdir(exist_ok=True)

		seen_images = set()

		with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["id_image", "annotation", "label"])

			for split in ["benign", "malignant", "normal"]:
				split_dir = nested_dir / split
				if split_dir.exists():
					for img_path in list(split_dir.iterdir()):
						if (
							img_path.is_file()
							and "Annotation" not in img_path.name
							and "mask" not in img_path.name
							and img_path.suffix == ".png"
						):
							if img_path.name in seen_images:
								continue

							label = "malignant" if split == "malignant" else "benign"

							if label == "malignant":
								annotation_name = (
									f"{img_path.stem}_Annotation{img_path.suffix}"
								)
								annotation_path = split_dir / annotation_name

								if annotation_path.exists():
									shutil.move(
										str(img_path), str(images_dir / img_path.name)
									)
									shutil.move(
										str(annotation_path),
										str(images_dir / annotation_name),
									)
									writer.writerow(
										[img_path.name, annotation_name, label]
									)
									seen_images.add(img_path.name)
							else:
								shutil.move(
									str(img_path), str(images_dir / img_path.name)
								)
								writer.writerow([img_path.name, "", label])
								seen_images.add(img_path.name)

		shutil.rmtree(data_dir / "Ultrasound Fetus Dataset", ignore_errors=True)
		shutil.rmtree(data_dir / ".complete", ignore_errors=True)
		(data_dir / "ultrasound_fetus.csv").unlink(missing_ok=True)
