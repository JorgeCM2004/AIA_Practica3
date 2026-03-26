from utils import Bounding_Boxes_Creator, Data_Downloader, Data_Splitter
from utils.loaders.F_Fetal_Ultrasound_Dataloader import Fetal_Ultrasound_Dataloader

SEED = 42  # None para no reproducibilidad.
BATCH_SIZE = 32


def main():
	downloader = Data_Downloader()
	downloader.download(force_download=False)
	bb_creator = Bounding_Boxes_Creator()
	bb_creator.create_YOLO()
	splitter = Data_Splitter()
	splitter.split(SEED)
	dataloader = Fetal_Ultrasound_Dataloader(batch_size=BATCH_SIZE)
	train_loader, test_loader = dataloader.create_dataloaders()


if __name__ == "__main__":
	main()
