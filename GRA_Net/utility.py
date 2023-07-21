import pathlib, time
import numpy as np
import tensorflow as tf
import pandas as pd

def merge_dataset() -> None:
	data_directory = pathlib.Path("/home/lorenzo/GazeDetection/GRA_Net/Output/train")
	image_count = len(list(data_directory.glob("*.npz")))

	output = None

	for file_path in data_directory.glob("*.npz"):
		try:
			data = np.load(file_path, allow_pickle = True)
			if output is None:
				output = data["arr_0"]
			else:
				output = np.concatenate((output, data["arr_0"]), axis = 0)
			data.close()
		except Exception as e:
			print(e)
			data.close()
			continue

	np.savez_compressed("/home/lorenzo/GazeDetection/GRA_Net/Output/Output", output)

def dataset_generator():
	with np.load("/home/lorenzo/GazeDetection/GRA_Net/Output/Output.npz", mmap_mode='r', allow_pickle = True) as data:
		for row in data["arr_0"]:
			yield row

def divide_images():
	dataset_dir = "/home/lorenzo/Dataset"
	image_count = len(list(pathlib.Path(dataset_dir).glob("*.jpg")))

	dir_num = int(image_count / 1000) + 1

	# Create directories
	for i in range(dir_num):
		pathlib.Path(dataset_dir + "/" + str(i)).mkdir(parents=True, exist_ok=True)

	shuffle = list(pathlib.Path(dataset_dir).glob("*.jpg"))
	np.random.shuffle(shuffle)

	# Move the images
	for i in range(image_count):
		pathlib.Path(shuffle[i]).rename(dataset_dir + "/" + str(int(i / 1000)) + "/" + shuffle[i].name)

def search_error():

	SOURCE_DATASET_PATH = "/home/lorenzo/Dataset/part1"
	# Load the dataset path
	data_directory = pathlib.Path(SOURCE_DATASET_PATH)
	print("Dataset path: " + str(data_directory))

	image_count = len(list(data_directory.glob("*.jpg")))
	print("Dataset size: " + str(image_count))

	# Load the dataset
	list_ds = tf.data.Dataset.list_files(str(data_directory/"*.jpg"), shuffle = False)

	for file in list_ds:
		try:
			image = tf.io.decode_image(tf.io.read_file(file), channels = 3)
			tf.print(filename = tf.strings.split(file, "/"))
			time.sleep(0.01)
		except Exception as e:
			print(e)
			print(file)
			continue

if __name__ == "__main__":

	search_error()
	#merge_dataset()
	#dataset_generator()
