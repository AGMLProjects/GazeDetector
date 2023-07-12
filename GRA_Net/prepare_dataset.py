import numpy as np
import pathlib, mlsocket, time, json
import tensorflow as tf
import tensorflow_datasets as tfds


if __name__ == "__main__":

	# Load the dataset path
	data_directory = pathlib.Path("/home/lorenzo/GazeDetection/Dataset/Gender")
	print("Dataset path: " + str(data_directory))

	image_count = len(list(data_directory.glob("*.jpg")))
	print("Dataset size: " + str(image_count))

	# Load the dataset
	list_ds = tf.data.Dataset.list_files(str(data_directory/"*.jpg"), shuffle = False)

	retrival_connector = mlsocket.MLSocket()
	retrival_connector.connect(("127.0.0.1", 65432))

	output = None
	i = 0
	y = 0

	for file_path in list_ds:

		while i < 10001:
			i = i + 1
			continue

		try:
			image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels = 3)
			image = image.numpy()
			label = [int(tf.strings.split(file_path, "_")[1]), int(int(tf.strings.split(file_path, "_")[2]) / 10) * 10]
		except Exception as e:
			print(e)
			continue

		time.sleep(0.001)
		retrival_connector.send(image)
		metadata = retrival_connector.recv(1024)
		metadata = json.loads(metadata.decode("utf-8"))

		if "status" not in metadata:
			raise KeyError("Missing status in the metadata")
		else:
			if metadata["status"] == True:
				embedding = retrival_connector.recv(1024)
				retrived_gender = int(metadata["gender"])
				retrived_age = int(int(metadata["age"]) / 10) * 10
				similarity = float(metadata["similarity"])
			else:
				embedding = tf.zeros(shape = (1), dtype = tf.uint8)
				retrived_gender, retrived_age = -1, -1
				similarity = -1

		row = [image, np.array(label, dtype = np.uint8), embedding, np.array([retrived_gender, retrived_age], dtype = np.uint8), np.array(similarity)]

		if output is None:
			output = np.array(row, dtype = object)
		else:
			output = np.append(output, np.array(row, dtype = object), axis = 0)

		if i % 1000 == 0 and i != 0:
			y = i / 1000
			np.savez_compressed(f"output{y}", output)
			output = None
			print(f"Saved output{y}.npy")
		
		i= i + 1

	