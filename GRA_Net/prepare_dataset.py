import numpy as np
import pathlib, mlsocket, hashlib, json
import tensorflow as tf

def get_label(file_path) -> int:
	gender = int(tf.strings.split(file_path, "_")[1])
	age = int(int(tf.strings.split(file_path, "_")[2]) / 10) * 10
	signature = str(hashlib.sha1(str(file_path).encode("UTF-8")).hexdigest().upper())

	return gender, age, signature

def decode_image(img) -> tf.Tensor:
	return tf.io.decode_jpeg(img, channels = 3)

def process_path(file_path) -> tuple:
	label = get_label(file_path = file_path)

	img = tf.io.read_file(file_path)
	img = decode_image(img = img)
	
	return img, label

if __name__ == "__main__":

	# Load the dataset path
	data_directory = pathlib.Path("/homes/lventurelli/Resources/Gender")
	print("Dataset path: " + str(data_directory))

	image_count = len(list(data_directory.glob("*.jpg")))
	print("Dataset size: " + str(image_count))

	# Load the dataset
	list_ds = tf.data.Dataset.list_files(str(data_directory/"*.jpg"), shuffle = False)

	retrival_connector = mlsocket.MLSocket()
	retrival_connector.connect(("127.0.0.1", 65432))

	list_ds = list_ds.map(process_path, num_parallel_calls = tf.data.experimental.AUTOTUNE)

	output = None

	for image, label in list_ds:
		x = image.numpy()
		retrival_connector.send(x)
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
				embedding = tf.zeros(shape = (1), dtype = tf.int16)
				retrived_gender, retrived_age = -1, -1
				similarity = -1

		row = [x, label[0].numpy(), label[1].numpy(), label[2].numpy(), embedding, np.array(retrived_gender), np.array(retrived_age), np.array(similarity)]

		if output is None:
			output = np.array(row, dtype = object)
		else:
			output = np.append(output, np.array(row), axis = 0)

		break

	print(output.shape)
	print(output[0])