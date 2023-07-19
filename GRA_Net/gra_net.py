from keras.layers import AveragePooling2D, Layer, Dropout, BatchNormalization, Conv2D, UpSampling2D, Activation, MaxPooling2D, Add, Multiply, Input, Dense, Flatten, Lambda
from keras.models import Model
from keras.regularizers import l2

import tensorflow as tf
import numpy as np
import pathlib, mlsocket, sys, json, random

class NumpyGenerator(tf.keras.utils.Sequence):

	def __init__(self, path: str, batch_size: int = 64, image_shape: tuple = (160, 160, 3), shuffle: bool = True):
		self._path = path
		self._batch_size = batch_size
		self._image_shape = image_shape

		self._image_file = self._path + "/images.npy"
		self._embedding_file = self._path + "/embeddings.npy"
		self._similarity_file = self._path + "/similarity.npy"
		self._gender_file = self._path + "/gender.npy"
		self._age_file = self._path + "/age.npy"
		self._ret_gender_file = self._path + "/retrived_gender.npy"
		self._ret_age_file = self._path + "/retrived_age.npy"

		self._opened = False
		self._shuffle = shuffle
		self._indices = list(range(0, self.__len__()))

		self._f_im, self._f_em, self._f_sim, self._f_g, self._f_a, self._f_rg, self._f_ra = None, None, None, None, None, None, None

	def __len__(self) -> int:
		t = np.memmap(self._image_file, dtype = "float32", mode = "r")
		i = int(t.shape[0] / int(self._image_shape[0] * self._image_shape[1] * self._image_shape[2] * self._batch_size))
		del t
		return i
		
	def __getitem__(self, idx):
		self._open_files()

		idx = idx % self.__len__()

		x = self._f_im[self._indices[idx] * self._batch_size:(self._indices[idx] + 1) * self._batch_size]
		x_ret = self._f_em[self._indices[idx] * self._batch_size:(self._indices[idx] + 1) * self._batch_size]
		sim = self._f_sim[self._indices[idx] * self._batch_size:(self._indices[idx] + 1) * self._batch_size]
		g = self._f_g[self._indices[idx] * self._batch_size:(self._indices[idx] + 1) * self._batch_size]
		a = np.clip(self._f_a[self._indices[idx] * self._batch_size:(self._indices[idx] + 1) * self._batch_size] / 10, 0, 9)
		g_ret = self._f_rg[self._indices[idx] * self._batch_size:(self._indices[idx] + 1) * self._batch_size]
		a_ret = np.clip(self._f_ra[self._indices[idx] * self._batch_size:(self._indices[idx] + 1) * self._batch_size] / 10, 0, 9)

		if self._batch_size > 1:
			return (
				{
					"img": tf.convert_to_tensor(x),
					"ret_img": tf.convert_to_tensor(x_ret),
					"ret_age": tf.convert_to_tensor(g_ret),
					"ret_gender": tf.convert_to_tensor(a_ret),
					"sim": tf.convert_to_tensor(sim)
				},
				{
					"gender": tf.convert_to_tensor(g),
					"age": tf.convert_to_tensor(a)
				}
			)
		else:
			return (
				{
					"img": tf.convert_to_tensor(x[0]),
					"ret_img": tf.convert_to_tensor(x_ret[0]),
					"ret_age": tf.convert_to_tensor(g_ret[0]),
					"ret_gender": tf.convert_to_tensor(a_ret[0]),
					"sim": tf.convert_to_tensor(sim[0])
				},
				{
					"gender": tf.convert_to_tensor(g[0]),
					"age": tf.convert_to_tensor(a[0])
				}
			)
	
	def on_epoch_end(self):
		if self._shuffle:
			random.shuffle(self._indices)
		
	def _open_files(self) -> None:
		if not self._opened:
			t = np.memmap(self._image_file, dtype = "float32", mode = "r")
			file_len = int(t.shape[0] / int(self._image_shape[0] * self._image_shape[1] * self._image_shape[2]))
			del t
			
			self._f_im = np.memmap(self._image_file, dtype = "float32", mode = "r", shape = (file_len, self._image_shape[0], self._image_shape[1], self._image_shape[2]))
			self._f_em = np.memmap(self._embedding_file, dtype = "float32", mode = "r", shape = (file_len, 1, 512))
			self._f_sim = np.memmap(self._similarity_file, dtype = "float64", mode = "r", shape = (file_len, 1))
			self._f_g = np.memmap(self._gender_file, dtype = "uint8", mode = "r", shape = (file_len, 1))
			self._f_a = np.memmap(self._age_file, dtype = "uint8", mode = "r", shape = (file_len, 1))
			self._f_rg = np.memmap(self._ret_gender_file, dtype = "uint8", mode = "r", shape = (file_len, 1))
			self._f_ra = np.memmap(self._ret_age_file, dtype = "uint8", mode = "r", shape = (file_len, 1))

			self._opened = True

class OnEpochCallback(tf.keras.callbacks.Callback):
	def __init__(self, generator: NumpyGenerator):
		super(OnEpochCallback, self).__init__()
		self._generator = generator

	def on_epoch_end(self, epoch, logs = None):
		print(f"Completed epoch {epoch}")
		self._generator.on_epoch_end()

class ModelTrainer():
	def __init__(self, dataset_path: str, batch_size: int, epochs: int, class_name: list, data_regex: str, n_classes: int):
		for item in [dataset_path, data_regex]:
			if not isinstance(item, str):
				raise TypeError("The dataset path and the data regex must be strings")
			
		for item in [batch_size, epochs, n_classes]:
			if not isinstance(item, int):
				raise TypeError("The batch size, the epochs and the n_classes must be integers")
			
		if not isinstance(class_name, list):
			raise TypeError("The class name must be a list")
		
		self._dataset_path = dataset_path
		self._batch_size = batch_size
		self._epochs = epochs
		self._class_name = class_name
		self._data_regex = data_regex
		self._n_classes = n_classes
		self._image_size = 64

	def _configure_dataset(self, ds):
		ds = ds.cache()
		ds = ds.batch(self._batch_size)
		ds = ds.prefetch(buffer_size = tf.data.AUTOTUNE)

		return ds
	
	def _numpy_generator(self, iterator: NumpyGenerator):
		for element in iterator:
			yield element

	def train_model(self):		
		dataset_generator = NumpyGenerator(path = self._dataset_path, batch_size = 1)
		on_epoch_callback = OnEpochCallback(dataset_generator)

		dataset = tf.data.Dataset.from_generator(
			lambda: dataset_generator, 
			output_signature = (
				{
					"img": tf.TensorSpec(shape = (160, 160, 3), dtype = tf.float32),
					"ret_img": tf.TensorSpec(shape = (1, 512), dtype = tf.float32),
					"ret_gender": tf.TensorSpec(shape = (1,), dtype = tf.uint8),
					"ret_age": tf.TensorSpec(shape = (1,), dtype = tf.uint8),
					"sim": tf.TensorSpec(shape = (1,), dtype = tf.float64)
				},
				{
					"gender": tf.TensorSpec(shape = (1,), dtype = tf.uint8),
					"age": tf.TensorSpec(shape = (1,), dtype = tf.uint8),
				}
			)
		)

		dataset = dataset.repeat()

		# Split the dataset in train and test
		train_size = int(0.8 * dataset_generator.__len__())
		train_ds = dataset.take(train_size)
		test_ds = dataset.skip(train_size)

		# Preprocessing steps
		rescale = tf.keras.Sequential([
			tf.keras.layers.Rescaling(1. / 255)
		])

		data_augmentation = tf.keras.Sequential([
			tf.keras.layers.RandomFlip("horizontal_and_vertical"),
			tf.keras.layers.RandomRotation(0.2),
			tf.keras.layers.RandomZoom(0.2),
			tf.keras.layers.RandomContrast(0.2),
			tf.keras.layers.RandomBrightness(0.2)
		])

		train_ds = train_ds.map(lambda x, y: ({"img": rescale(x["img"]), "ret_img": x["ret_img"], "ret_gender": x["ret_gender"], "ret_age": x["ret_age"], "sim": x["sim"]}, y), num_parallel_calls = tf.data.AUTOTUNE)
		train_ds = train_ds.concatenate(train_ds.map(lambda x, y: ({"img": data_augmentation(x["img"], training = True), "ret_img": x["ret_img"], "ret_gender": x["ret_gender"], "ret_age": x["ret_age"], "sim": x["sim"]}, y), num_parallel_calls = tf.data.AUTOTUNE))

		test_ds = test_ds.map(lambda x, y: ({"img": rescale(x["img"]), "ret_img": x["ret_img"], "ret_gender": x["ret_gender"], "ret_age": x["ret_age"], "sim": x["sim"]}, y), num_parallel_calls = tf.data.AUTOTUNE)


		# Optimize the dataset
		train_ds = self._configure_dataset(train_ds)
		test_ds = self._configure_dataset(test_ds)

		# Define the model
		model = GenderNetwork(shape = (self._image_size, self._image_size, 3), n_channels = 64, n_classes = self._n_classes, dropout = 0.1, regularization = 0.01)

		optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, weight_decay = 0.0001, use_ema = True, ema_momentum = 0.9, clipnorm = 1.0)
		loss = {"gender": tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), "age": tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)}
		metrics = {"gender": tf.keras.metrics.SparseCategoricalAccuracy(name = "Gender Accuracy"), "age": tf.keras.metrics.SparseCategoricalAccuracy(name = "Age Accuracy")}

		# Compile the model
		model.compile(
			optimizer = optimizer,
			loss = loss,
			metrics = metrics,
			run_eagerly = False
		)


		model.summary()
		
		# Train the model
		model.fit(
			train_ds,
			steps_per_epoch = int(train_size / self._batch_size),
			validation_data = test_ds,
			shuffle = True,
			epochs = EPOCHS,
			callbacks = [on_epoch_callback]
		)

		# Print the accuracy
		print("Accuracy: {}".format(model.evaluate(test_ds)[1]))

		model.save("GraNet.h5")

# Define the GatedLayer class which will be define both the residual and the gated blocks
class GatedLayer(Layer):

	def __init__(self, **kwargs):
		super(GatedLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create trainable weights for this layer
		self._gamma = self.add_weight(name = 'gamma', shape = (1,), initializer = 'uniform', trainable = True)
		super(GatedLayer, self).build(input_shape)

	def call(self, x):
		return Add()([
			self._gamma, 
			Multiply()([
				1 - self._gamma, 
				x
			])
		])
	
	def compute_output_shape(self, input_shape):
		return input_shape[0]

# Define the CombinationGate class which will be used to combine the embedding of the current image with the embedding of the retrieved image
class CombinationGate(Layer):
	def __init__(self, **kwargs):
		super(CombinationGate, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create trainable weights for this layer
		self.alpha = self.add_weight(name = 'alpha', shape = (1,), initializer = 'uniform', trainable = True, dtype = tf.float32)
		super(CombinationGate, self).build(input_shape[0])

	def call(self, x, x_ret, sim):
		s = tf.keras.backend.cast(sim >= 0.5, dtype = tf.float32)

		return Add()([
			Multiply()([
				self.alpha,
				pow(1, s),
				x
			]),
			Multiply()([
				(1 - self.alpha),
				s,
				x_ret
			])
		])
	
	def compute_output_shape(self, input_shape):
		return input_shape

# Define the residual block
def residual_block(input, input_channels = None, output_channels = None, kernel_size = (3, 3), stride = 1):

	# Shape the channels according to the parameters
	if output_channels is None:
		output_channels = input.get_shape()[-1]

	if input_channels is None:
		input_channels = output_channels // 4

	# Define the stride in 2D
	strides = (stride, stride)

	# Define the forward pass of the residual block
	x = BatchNormalization()(input)
	x = Activation('relu')(x)
	x = Conv2D(input_channels, (1, 1), strides = (1, 1))(x)         # 1x1 convolution with stride 1

	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(input_channels, kernel_size, strides = strides, padding = 'same')(x)    # 3x3 convolution with stride 1 and padding with zeroes

	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(output_channels, (1, 1), padding = "same")(x)        # 1x1 convolution with stride 1 and padding with zeroes

	# If the input channels and the output channels are different, we need to perform a convolution on the input in order to match the dimensions
	if input_channels != output_channels or stride != 1:
		input = Conv2D(output_channels, (1, 1), strides = strides, padding = 'same')(input)

	# Add the input to the output of the residual block
	x = Add()([x, input])
	
	return x

# Define the attention block
def attention_block(input, input_channels = None, output_channels = None, encoder_depth = 1):

	# Define the parameters of the attention block
	p = 1
	t = 2
	r = 1

	# Shape the channels according to the parameters
	if input_channels is None:
		input_channels = input.get_shape()[-1]

	if output_channels is None:
		output_channels = input_channels

	# First residual block
	for i in range(p):
		input = residual_block(input = input)

	# Trunk branch
	output_trunk = input
	for i in range(t):
		output_trunk = residual_block(input = output_trunk)

	# Soft mask branch
	# Encoder
	# First downsampling
	output_soft_mask = MaxPooling2D(padding = "same")(input)        # 32x32 Pooling

	for i in range(r):
		output_soft_mask = residual_block(input = output_soft_mask)

	# Define the skip connections
	skip_connections = []
	for i in range(encoder_depth - 1):
		output_skip_connection = residual_block(input = output_soft_mask)
		skip_connections.append(output_skip_connection)

		# Downsampling
		output_soft_mask = MaxPooling2D(padding = "same")(output_soft_mask)

		for _ in range(r):
			output_soft_mask = residual_block(input = output_soft_mask)

	# Decoder
	skip_connections = list(reversed(skip_connections))
	for i in range(encoder_depth - 1):
		# Upsampling
		for _ in range(r):
			output_soft_mask = residual_block(input = output_soft_mask)
		
		output_soft_mask = UpSampling2D()(output_soft_mask)
		
		# Skip connection
		output_soft_mask = Add()([output_soft_mask, skip_connections[i]])
	
	# Last upsampling
	for i in range(r):
		output_soft_mask = residual_block(input = output_soft_mask)

	output_soft_mask = UpSampling2D()(output_soft_mask)

	# Output
	output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
	output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
	output_soft_mask = Activation('sigmoid')(output_soft_mask)

	# Attention
	output = GatedLayer()(output_soft_mask)
	output = Lambda(lambda x: x + 1)(output_soft_mask)
	output = Multiply()([output, output_trunk])

	# Last residual block
	for i in range(p):
		output = residual_block(input = output)

	return output

# Define the model
def GenderNetwork(shape: tuple, n_channels: int, n_classes: int, dropout: float, regularization: float):
	for item in [n_channels, n_classes]:
		if not isinstance(item, int):
			raise TypeError("The shape, the n_channels and the n_classes must be integers")
		
	for item in [dropout, regularization]:
		if not isinstance(item, float):
			raise TypeError("The dropout and the regularization must be floats")
		
	if not isinstance(shape, tuple) and len(shape) != 3:
		raise TypeError("The shape must be a tuple of 3 elements")

	# Define the regularization factor
	regularizer = l2(regularization)

	# Define the input layers
	input_ = Input(shape = (160, 160, 3), name = "img", dtype = tf.float32)
	sim_ = Input(shape = (), name = "sim", dtype = tf.float64)
	x_ret_ = Input(shape = (1, 512), name = "ret_img", dtype = tf.float32)
	gender_ret_ = Input(shape = (), name = "ret_gender", dtype = tf.uint8)
	age_ret_ = Input(shape = (), name = "ret_age", dtype = tf.uint8)

	# Perform preliminar normalization, reshaping and pooling before feeding the attention network
	x = Conv2D(n_channels, (7, 7), strides = (2, 2), padding = "same")(input_)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same")(x)

	# Feed the GRA_Net
	x = attention_block(x, encoder_depth=3)                             # 56x56
	x = residual_block(x, output_channels=n_channels * 4)               # Bottleneck 7x7

	x = attention_block(x, encoder_depth=2)                             # 28x28
	x = residual_block(x, output_channels=n_channels * 8, stride=2)     # Bottleneck 7x7

	x = attention_block(x, encoder_depth=1)                             # 14x14
	x = residual_block(x, output_channels=n_channels * 16, stride=2)    # Bottleneck 7x7
	
	x = residual_block(x, output_channels=n_channels * 32)
	x = residual_block(x, output_channels=n_channels * 32)
	x = residual_block(x, output_channels=n_channels * 32)

	# Final pooling and softmax steps
	pool_size = (x.get_shape()[1], x.get_shape()[2])
	x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
	x = Flatten()(x)
	x_ret = Flatten()(x_ret_)
	sim = tf.cast(sim_, tf.float32)

	# First Dense Layer
	x = Dense(n_channels * 32, kernel_regularizer = regularizer, activation = "relu")(x)

	# Second Dense Layer: map the embedding to the fixed shape of 512 features
	x = Dense(512, kernel_regularizer = regularizer, activation = "relu")(x)

	# Combination gate
	x = CombinationGate()(x, x_ret, sim)

	# Apply dropout if needed
	if dropout:
		x = Dropout(dropout)(x)

	# Divide the prediction in two branches
	
	# Gender branch
	y_ret_gender = tf.one_hot(gender_ret_, depth = 2)
	x_gender = Dense(2, kernel_regularizer = regularizer, activation = "softmax")(tf.keras.layers.Concatenate(axis = 1)([x, y_ret_gender]))

	# Age branch
	y_ret_age = tf.one_hot(age_ret_, depth = 10)
	x_age = Dense(10, kernel_regularizer = regularizer, activation = "softmax")(tf.keras.layers.Concatenate(axis = 1)([x, y_ret_age]))

	model =  Model(
		{
			"img": input_, 
			"sim": sim_, 
			"ret_img": x_ret_, 
			"ret_gender": gender_ret_,
			"ret_age": age_ret_
		}, 
		{
			"gender": x_gender, 
			"age": x_age
		}
	)
	return model

if __name__ == "__main__":

	# Define the constants
	DATASET_PATH = "./Gender/"
	BATCH_SIZE = 64
	EPOCHS = 20
	CLASS_NAME = ["female", "male"]
	DATA_REGEX = r"([0-9]+)[\_]+([0-9]+)[\_]+([0-9]+)[\_]+([0-9]+)\.jpg"

	if len(sys.argv) < 2:
		print("Usage: [train|inference] {[dataset_path] [gender|age]}")
		sys.exit(1)
	else:
		mode = sys.argv[1]

		if mode not in ["train", "inference"]:
			print("Usage: [train|inference] {[dataset_path] [gender|age]}")
			sys.exit(1)

		if mode == "train":
			if len(sys.argv) < 4:
				print("If used in train mode: [train|inference] [dataset_path] [gender|age]")
			else:
				dataset_path = sys.argv[2]
				train_type = sys.argv[3]

				if train_type not in ["gender", "age"]:
					print("Usage: [train|inference] {[dataset_path] [gender|age]}")
					sys.exit(1)

	if mode == "train":
		if train_type == "gender":
			trainer = ModelTrainer(dataset_path = dataset_path, batch_size = BATCH_SIZE, epochs = EPOCHS, class_name = CLASS_NAME, data_regex = DATA_REGEX, n_classes = 2)
		else:
			trainer = ModelTrainer(dataset_path = dataset_path, batch_size = BATCH_SIZE, epochs = EPOCHS, class_name = CLASS_NAME, data_regex = DATA_REGEX, n_classes = 10)

		trainer.train_model()
		sys.exit(0)
	else:
		camera_socket = mlsocket.MLSocket()
		camera_socket.bind(("0.0.0.0", 12345))
		camera_socket.listen()

		model = GenderNetwork(shape = (160, 160, 3), n_channels = 64, n_classes = 2, dropout = 0.1, regularization = 0.01)
		model.load_weights("GraNet.h5")

		retrival_connector = mlsocket.MLSocket()
		retrival_connector.connect(("127.0.0.1", 65432))

		print("Waiting for a connection...")

		camera, address = camera_socket.accept()
		print("Connection established with", address)

		with camera:
			while True:
				data = camera.recv(1024)
				if data == b"":
					break

				retrival_connector.send(data)
				metadata = retrival_connector.recv(1024)
				metadata = json.loads(metadata.decode("utf-8"))

				if "status" not in metadata:
					raise KeyError("Missing status in the metadata")
				else:
					if metadata["status"] == True:
						embedding = retrival_connector.recv(1024)
						inputs = {
							"img" : data,
							"sim" : metadata["similarity"],
							"ret_img" : embedding,
							"ret_gender": metadata["gender"],
							"ret_age": metadata["age"]
						}

						prediction = model.predict(inputs)

						camera.send(json.dumps({"status": False, "age": prediction["age"], "gender": prediction["gender"]}).encode("UTF-8"))
					else:
						camera.send(json.dumps({"status": False, "age": 0, "gender": 0}).encode("UTF-8"))





