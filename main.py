from utils import (
	Bounding_Boxes_Creator,
	Classifier_Manager,
	Data_Downloader,
	Data_Splitter,
	Fetal_Ultrasound_Dataloader,
	Metrics_Generator,
	YOLO_Model,
)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20


def main():
	downloader = Data_Downloader()
	downloader.download(force_download=False)

	bb_creator = Bounding_Boxes_Creator()
	bb_creator.create_YOLO()

	splitter = Data_Splitter()
	splitter.split(SEED)

	dataloader = Fetal_Ultrasound_Dataloader(batch_size=BATCH_SIZE)
	train_loader, test_loader = dataloader.create_dataloaders()

	manager = Classifier_Manager(models="All", seed=SEED)
	manager.train(data_loader=train_loader, epochs=EPOCHS)
	out = manager.predict(test_loader)

	metrics = Metrics_Generator()
	for name, data in out.items():
		y_val_true, y_val_hat = data
		metrics.generate_clasification(name, y_val_true, y_val_hat)

	yolo = YOLO_Model()

	yolo.train(
		epochs=5,
		batch_size=32,
	)

	predicciones = yolo.predict(source="Dataset_YOLO/images/val")

	metrics.generate_detection(model_name="YOLO", y_hat=predicciones)


if __name__ == "__main__":
	main()
