import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class Bounding_Boxes_Creator:
	def __init__(self):
		self.data_dir = Path(__file__).parent.parent.parent / "data"
		self.images_dir = self.data_dir / "images"
		self.csv_path = self.data_dir / "fetal_ultrasound.csv"

	def create_YOLO(self):
		df = pd.read_csv(self.csv_path)

		if "yolo_file" not in df.columns:
			df["yolo_file"] = None

		df_malignant = df[df["label"] == "malignant"]

		yolo_dir = self.data_dir / "YOLO_bounding_boxes"

		shutil.rmtree(yolo_dir, ignore_errors=True)
		yolo_dir.mkdir(exist_ok=True)

		for index, row in df_malignant.iterrows():
			img_name = row["id_image"]
			annotation_name = row["annotation"]

			if pd.isna(annotation_name) or not annotation_name:
				continue

			anotation_path = self.images_dir / annotation_name

			if not anotation_path.exists():
				continue

			mask = cv2.imread(str(anotation_path), cv2.IMREAD_GRAYSCALE)
			if mask is None:
				continue

			y_indices, x_indices = np.where(mask > 0)

			if len(x_indices) == 0 or len(y_indices) == 0:
				continue

			x_min, x_max = np.min(x_indices), np.max(x_indices)
			y_min, y_max = np.min(y_indices), np.max(y_indices)

			h_img, w_img = mask.shape

			x_center = ((x_min + x_max) / 2.0) / w_img
			y_center = ((y_min + y_max) / 2.0) / h_img
			width = (x_max - x_min) / w_img
			height = (y_max - y_min) / h_img

			txt_name = Path(img_name).stem + ".txt"
			txt_path = yolo_dir / txt_name

			with open(txt_path, "w", encoding="utf-8") as f:
				f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

			df.at[index, "yolo_file"] = txt_name

		df = df[["id_image", "annotation", "yolo_file", "label"]]
		df.to_csv(self.csv_path, index=False)
