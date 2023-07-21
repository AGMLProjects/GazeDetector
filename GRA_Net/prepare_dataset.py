import numpy as np
import pathlib, mlsocket, time, json
import tensorflow as tf


if __name__ == "__main__":

	BATCH_SIZE = 256
	SOURCE_DATASET_PATH = "/home/lorenzo/Dataset/part1"
	DESTINATION_DATASET_PATH = "/home/lorenzo/GazeDetection/GRA_Net/Dataset/train"

	# Load the dataset path
	data_directory = pathlib.Path(SOURCE_DATASET_PATH)
	print("Dataset path: " + str(data_directory))

	image_count = len(list(data_directory.glob("*.jpg")))
	print("Dataset size: " + str(image_count))

	# Load the dataset
	list_ds = tf.data.Dataset.list_files(str(data_directory/"*.jpg"), shuffle = True)

	retrival_connector = mlsocket.MLSocket()
	retrival_connector.connect(("127.0.0.1", 65432))

	if pathlib.Path(f"{DESTINATION_DATASET_PATH}/images.npy").exists():
		t = np.memmap(f"{DESTINATION_DATASET_PATH}/images.npy", dtype = "float32", mode = "r")
		skip_images = int(t.shape[0] / (160*160*3))
		completed_steps = int(skip_images / BATCH_SIZE)
		del t
	else:
		skip_images = 0
		completed_steps = 0

	print(f"Skipping the first {skip_images} images. Starting from batch {completed_steps}/{int(image_count / BATCH_SIZE)}")

	x, x_ret, g, a, g_ret, a_ret, s = None, None, None, None, None, None, None
	x_shape, x_ret_shape, g_shape, a_shape, g_ret_shape, a_ret_shape, s_hape = (160, 160, 3), (1, 512), (1,), (1,), (1,), (1,), (1,)
	shape = (160, 160, 3)
	i = 1
	y = 0

	for file_path in list_ds:

		if i <= skip_images:
			if i == skip_images:
				y = completed_steps

			i = i + 1
			continue

		try:
			image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels = 3)
			image = tf.image.resize(image, [160, 160], antialias = True)
			image = image.numpy()
			filename = tf.strings.split(file_path, "/")[-1]
			label = [int(tf.strings.split(filename, "_")[1]), int(int(tf.strings.split(filename, "_")[0]) / 10) * 10]
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
				shape = embedding.shape
				retrived_gender = int(metadata["gender"])
				retrived_age = int(int(metadata["age"]) / 10) * 10
				similarity = float(metadata["similarity"])
			else:
				continue

		if x is None:
			x = np.array((image,), dtype = np.float32)
			x_ret = np.array((embedding,), dtype = np.float32)
			g = np.array((label[0],), dtype = np.uint8)
			a = np.array((label[1],), dtype = np.uint8)
			g_ret = np.array((retrived_gender,), dtype = np.uint8)
			a_ret = np.array((retrived_age,), dtype = np.uint8)
			s = np.array((similarity,), dtype = np.float64)
		else:
			x = np.append(x, np.array((image,), dtype = np.float32), axis = 0)
			x_ret = np.append(x_ret, np.array((embedding,), dtype = np.float32), axis = 0)
			g = np.append(g, np.array((label[0],), dtype = np.uint8), axis = 0)
			a = np.append(a, np.array((label[1],), dtype = np.uint8), axis = 0)
			g_ret = np.append(g_ret, np.array((retrived_gender,), dtype = np.uint8), axis = 0)
			a_ret = np.append(a_ret, np.array((retrived_age,), dtype = np.uint8), axis = 0)
			s = np.append(s, np.array((similarity,), dtype = np.float64), axis = 0)

		if i % BATCH_SIZE == 0 and i != 0:
			if y == 0:
				f = np.memmap(f"{DESTINATION_DATASET_PATH}/images.npy", dtype = "float32", mode = "w+", shape = (i, x_shape[0], x_shape[1], x_shape[2]))
				f[:, :, :, :] = x[:, :, :, :]
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/embeddings.npy", dtype = "float32", mode = "w+", shape = (i, x_ret_shape[0], x_ret_shape[1]))
				f[:, :, :] = x_ret[:, :, :]
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/gender.npy", dtype = "uint8", mode = "w+", shape = (i,))
				f[:] = g[:]
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/age.npy", dtype = "uint8", mode = "w+", shape = (i,))
				f[:] = a[:]
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/retrived_gender.npy", dtype = "uint8", mode = "w+", shape = (i,))
				f[:] = g_ret[:]
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/retrived_age.npy", dtype = "uint8", mode = "w+", shape = (i,))
				f[:] = a_ret[:]
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/similarity.npy", dtype = "float64", mode = "w+", shape = (i,))
				f[:] = s[:]
				f.flush()
				del f

			else:
				f = np.memmap(f"{DESTINATION_DATASET_PATH}/images.npy", dtype = "float32", mode = "r+", shape = (i, x_shape[0], x_shape[1], x_shape[2]))
				f[i - BATCH_SIZE:, :, :, :] = x
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/embeddings.npy", dtype = "float32", mode = "r+", shape = (i, x_ret_shape[0], x_ret_shape[1]))
				f[i - BATCH_SIZE:, :, :] = x_ret
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/gender.npy", dtype = "uint8", mode = "r+", shape = (i,))
				f[i - BATCH_SIZE:] = g
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/age.npy", dtype = "uint8", mode = "r+", shape = (i,))
				f[i - BATCH_SIZE:] = a
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/retrived_gender.npy", dtype = "uint8", mode = "r+", shape = (i,))
				f[i - BATCH_SIZE:] = g_ret
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/retrived_age.npy", dtype = "uint8", mode = "r+", shape = (i,))
				f[i - BATCH_SIZE:] = a_ret
				f.flush()
				del f

				f = np.memmap(f"{DESTINATION_DATASET_PATH}/similarity.npy", dtype = "float64", mode = "r+", shape = (i,))
				f[i - BATCH_SIZE:] = s
				f.flush()
				del f
			
			x, x_ret, g, a, g_ret, a_ret, s = None, None, None, None, None, None, None
			y = int(i / BATCH_SIZE)
			print(f"Calculated batch {y}/{int(image_count / BATCH_SIZE)}")
			
		
		i = i + 1