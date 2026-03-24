from utils import Data_Downloader, Data_Splitter

SEED = 42  # None para no reproducibilidad.


def main():
	downloader = Data_Downloader()
	downloader.download(force_download=False)
	splitter = Data_Splitter()
	splitter.split(SEED)


if __name__ == "__main__":
	main()
