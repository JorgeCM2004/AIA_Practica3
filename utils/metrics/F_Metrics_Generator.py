import shutil
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	f1_score,
	precision_score,
	recall_score,
)


class Metrics_Generator:
	def __init__(self):
		self.results = Path(__file__).parent.parent.parent / "results"

	def generate_clasification(self, model_name, y_true, y_hat):
		if (self.results / model_name).exists():
			shutil.rmtree(self.results / model_name)
		(self.results / model_name).mkdir(parents=True, exist_ok=True)
		self.quantifiable_metrics(model_name, y_true, y_hat)
		self.confussion_matrix_classification(model_name, y_true, y_hat)

	def generate_detection(
		self, model_name, y_hat, val_images_dir="Dataset_YOLO/images/val"
	):
		res_dir = self.results / model_name
		res_dir.mkdir(parents=True, exist_ok=True)

		img_dir = Path(val_images_dir)
		val_labels_dir = img_dir.parent.parent / "labels" / img_dir.name
		ann_dir = Path("data/images")

		val_imgs = sorted(list(img_dir.glob("*.png")))

		for img_path, p_str in zip(val_imgs, y_hat):
			t_str = ""
			txt_path = val_labels_dir / img_path.with_suffix(".txt").name
			if txt_path.exists():
				with open(txt_path, "r") as f:
					t_str = f.readline().strip()

			img = cv2.imread(str(img_path))
			if img is None:
				continue
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			h, w = img.shape[:2]

			fig, ax = plt.subplots(figsize=(8, 8))
			ax.imshow(img)
			ax.axis("off")

			def get_rect(box_str, color, label):
				if not box_str or box_str.strip() == "":
					return None
				parts = str(box_str).strip().split()
				if len(parts) < 5:
					return None
				_, xc, yc, bw, bh = map(float, parts[:5])
				if bw == 0 or bh == 0:
					return None
				return patches.Rectangle(
					((xc - bw / 2) * w, (yc - bh / 2) * h),
					bw * w,
					bh * h,
					linewidth=2,
					edgecolor=color,
					facecolor="none",
					label=label,
				)

			legend_handles = []

			p_rect = get_rect(p_str, "blue", "Y_HAT")
			if p_rect:
				ax.add_patch(p_rect)
				legend_handles.append(p_rect)

			t_rect = get_rect(t_str, "green", "Y_TRUE")
			if t_rect:
				ax.add_patch(t_rect)
				legend_handles.append(t_rect)

			ann_path = ann_dir / (img_path.stem + "_Annotation.png")
			if ann_path.exists():
				mask = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
				if mask is not None:
					if len(mask.shape) > 2:
						mask = mask[:, :, 0]

					_, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
					contours, _ = cv2.findContours(
						thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
					)

					if contours:
						c = max(contours, key=cv2.contourArea)
						pts = c.reshape(-1, 2)

						if len(pts) >= 3:
							poly = patches.Polygon(
								pts,
								closed=True,
								linewidth=2,
								edgecolor="red",
								facecolor="none",
								label="ANNOTATION",
							)
							ax.add_patch(poly)
							legend_handles.append(poly)

			if legend_handles:
				ax.legend(
					handles=legend_handles,
					loc="upper left",
					bbox_to_anchor=(1, 1),
					frameon=False,
				)

			plt.savefig(res_dir / img_path.name, bbox_inches="tight", dpi=150)
			plt.close()

	def quantifiable_metrics(self, model_name, y_true, y_hat):
		with open((self.results / model_name / "metrics.txt").resolve(), "w") as file:
			file.write(f"Accuracy: {accuracy_score(y_true, y_hat)}\n")
			file.write(f"Recall: {recall_score(y_true, y_hat, average='weighted')}\n")
			file.write(
				f"Precision: {precision_score(y_true, y_hat, average='weighted')}\n"
			)
			file.write(f"F1-Score: {f1_score(y_true, y_hat, average='weighted')}\n")
			file.write(f"Macro F1-Score: {f1_score(y_true, y_hat, average='macro')}\n")

	def confussion_matrix_classification(self, model_name, y_true, y_hat):
		ConfusionMatrixDisplay.from_predictions(y_true, y_hat)
		plt.savefig((self.results / model_name / "confussion_matrix.png").resolve())
		plt.close()
