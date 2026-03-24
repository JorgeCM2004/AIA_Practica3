import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class Data_Splitter:
	def __init__(self):
		self.data_folder = Path(__file__).parent.parent.parent / "data"
		self.csv_path = self.data_folder / "fetal_ultrasound.csv"
		self.images_folder = self.data_folder / "images"

	def clean(self):
		if not self.csv_path.exists():
			raise ValueError("Debes descargar el dataset primero.")

		df = pd.read_csv(self.csv_path.resolve())

		anotations = df["id_image"].str.contains("Annotation", case=False, na=False)
		df_clean = df[~anotations]

		return df_clean

	def split(self, seed=None, test_size=0.2):
		if seed and isinstance(seed, int):
			random.seed(seed)

		df_clean = self.clean()

		(self.data_folder / "fetal_ultrasound_train.csv").unlink(missing_ok=True)
		(self.data_folder / "fetal_ultrasound_test.csv").unlink(missing_ok=True)

		train_df, test_df = train_test_split(
			df_clean, test_size=test_size, random_state=seed, stratify=df_clean["label"]
		)

		train_df.to_csv(
			(self.data_folder / "fetal_ultrasound_train.csv").resolve(), index=False
		)
		test_df.to_csv(
			(self.data_folder / "fetal_ultrasound_test.csv").resolve(), index=False
		)

		return train_df, test_df
