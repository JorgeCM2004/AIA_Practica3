import os
import shutil

import pandas as pd
from ultralytics import YOLO


class YOLO_Model:
	def __init__(self, model_version="yolov8n.pt", project_dir="YOLO_Localizer"):
		self.model = YOLO(model_version)
		self.project_dir = project_dir
		self.yalm_path = self._prepare_dataset()

	def _copy_data(self, df, split, base_dir, img_dir, bbox_dir):
		for _, row in df.iterrows():
			img_src = os.path.join(img_dir, row["id_image"])
			txt_src = os.path.join(bbox_dir, str(row["yolo_file"]))

			img_dst = os.path.join(base_dir, "images", split, row["id_image"])
			txt_name = row["id_image"].replace(".png", ".txt").replace(".jpg", ".txt")
			txt_dst = os.path.join(base_dir, "labels", split, txt_name)

			if os.path.exists(img_src) and os.path.exists(txt_src):
				shutil.copy(img_src, img_dst)
				shutil.copy(txt_src, txt_dst)

	def _prepare_dataset(
		self,
		base_dir="Dataset_YOLO",
	):
		df_train = pd.read_csv("data/fetal_ultrasound_train.csv")
		df_test = pd.read_csv("data/fetal_ultrasound_test.csv")

		df_train = df_train[df_train["label"] == "malignant"]
		df_test = df_test[df_test["label"] == "malignant"]

		folders = [
			f"{base_dir}/images/train",
			f"{base_dir}/labels/train",
			f"{base_dir}/images/val",
			f"{base_dir}/labels/val",
		]
		for folder in folders:
			os.makedirs(folder, exist_ok=True)

		self._copy_data(
			df_train, "train", "Dataset_YOLO", "data/images", "data/YOLO_bounding_boxes"
		)
		self._copy_data(
			df_test, "val", "Dataset_YOLO", "data/images", "data/YOLO_bounding_boxes"
		)

		yaml_path = f"{base_dir}/fetal_ultrasound.yaml"
		with open(yaml_path, "w") as f:
			f.write(f"train: {os.path.abspath(base_dir)}/images/train\n")
			f.write(f"val: {os.path.abspath(base_dir)}/images/val\n")
			f.write("nc: 1\n")
			f.write("names: ['malignant']\n")

		return yaml_path

	def train(
		self,
		epochs=10,
		batch_size=8,
		img_size=640,
	):
		results = self.model.train(
			data=self.yalm_path,
			epochs=epochs,
			imgsz=img_size,
			batch=batch_size,
			device=0,
			project=self.project_dir,
			exist_ok=True,
			workers=0,
			verbose=False,
		)
		return results

	def predict(self, source):
		results = self.model.predict(
			source=source,
			conf=0.000001,
			save=False,
			project=self.project_dir,
			name="predict_run",
			exist_ok=True,
			verbose=False,
		)

		y_pred_boxes = []
		for r in results:
			if len(r.boxes) > 0:
				best_box = r.boxes[0]
				x, y, w, h = best_box.xywhn[0].tolist()
				y_pred_boxes.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
			else:
				y_pred_boxes.append("0 0.000000 0.000000 0.000000 0.000000")
		return y_pred_boxes
