from keras.layers import AveragePooling2D, Layer, Dropout, BatchNormalization, Conv2D, UpSampling2D, Activation, MaxPooling2D, Add, Multiply, Input, Dense, Flatten, Lambda
from keras.models import Model
from keras.regularizers import l2

import tensorflow as tf
import numpy as np
import pathlib, mlsocket, sys, json

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

		self._retrival_connector = mlsocket.MLSocket()
		self._retrival_connector.connect(("127.0.0.1", 65432))

	def _get_label(self, file_path):
		if self._n_classes == 2:
			return int(tf.strings.split(file_path, "_")[1])
		else:
			return int(int(tf.strings.split(file_path, "_")[2]) / 10) * 10

	def _decode_image(self, img):
		return tf.io.decode_jpeg(img, channels = 3)

	def _process_path(self, file_path):
		label = self._get_label(file_path = file_path)

		img = tf.io.read_file(file_path)
		img = self._decode_image(img = img)
		
		return img, label

	def _configure_dataset(self, ds):
		ds = ds.cache()
		ds = ds.shuffle(buffer_size = 1000)
		ds = ds.batch(32)
		ds = ds.prefetch(buffer_size = tf.data.AUTOTUNE)

		return ds
	
	def _get_retrival(self, x, y):
		self._retrival_connector.send(x.numpy())
		metadata = self._retrival_connector.recv(1024)
		metadata = json.loads(metadata.decode("utf-8"))

		if "status" not in metadata:
			raise KeyError("Missing status in the metadata")
		else:
			if metadata["status"] == True:
				embedding = self._retrival_connector.recv(1024)
			else:
				embedding = None

		return metadata, embedding
	
	def train_model(self):
		# Load the dataset path
		data_directory = pathlib.Path(self._dataset_path)
		print("Dataset path: " + str(data_directory))

		image_count = len(list(data_directory.glob("*.jpg")))
		print("Dataset size: " + str(image_count))

		# Load the dataset
		list_ds = tf.data.Dataset.list_files(str(data_directory/"*.jpg"), shuffle = False)
		list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration = False)
		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		list_ds = list_ds.with_options(options)

		# Split the dataset into training and test
		val_size = int(image_count * 0.2)
		train_ds = list_ds.skip(val_size)
		test_ds = list_ds.take(val_size)

		# Define the preprocessing steps
		resize = tf.keras.Sequential([
			tf.keras.layers.Resizing(self._image_size, self._image_size)
		])

		rescale = tf.keras.Sequential([
			tf.keras.layers.Rescaling(1. / 255)
		])

		data_augmentation = tf.keras.Sequential([
			tf.keras.layers.RandomFlip("horizontal_and_vertical"),
			tf.keras.layers.RandomRotation(0.2),
			tf.keras.layers.RandomZoom(0.2),
			tf.keras.layers.RandomContrast(0.2)
		])

		# Decode the dataset and retrive the labels
		train_ds = train_ds.map(self._process_path, num_parallel_calls = tf.data.AUTOTUNE)
		test_ds = test_ds.map(self._process_path, num_parallel_calls = tf.data.AUTOTUNE)


		# TODO: Fare il retrival ed integrare il risultato nel dataset (prima di rescale e augmentation) sia per il train che per il test
		# Apply the preprocessing steps
		train_ds = train_ds.map(lambda x, y: (resize(x), y), num_parallel_calls = tf.data.AUTOTUNE)
		train_ds = train_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls = tf.data.AUTOTUNE)
		train_ds = train_ds.concatenate(train_ds.map(lambda x, y: (data_augmentation(x, training = True), y), num_parallel_calls = tf.data.AUTOTUNE))

		test_ds = test_ds.map(lambda x, y: (resize(x), y), num_parallel_calls = tf.data.AUTOTUNE)
		test_ds = test_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls = tf.data.AUTOTUNE)

		print("Train dataset size: " + str(len(train_ds)))
		print("Test dataset size: " + str(len(test_ds)))

		# Optimize the dataset
		train_ds = self._configure_dataset(train_ds)
		test_ds = self._configure_dataset(test_ds)

		# Define the model
		model = GenderNetwork(shape = (self._image_size, self._image_size, 3), n_channels = 64, n_classes = self._n_classes, dropout = 0.1, regularization = 0.01)

		optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, weight_decay = 0.0001, use_ema = True, ema_momentum = 0.9, clipnorm = 1.0)
		loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
		metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]


		# Compile the model
		model.compile(
			optimizer = optimizer,
			loss = loss,
			metrics = metrics
		)


		model.summary()
		
		# Train the model
		model.fit(
			train_ds,
			validation_data = test_ds,
			epochs = EPOCHS
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

	# Perform preliminar normalization, reshaping and pooling before feeding the attention network
	input_ = Input(shape = shape)
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
	if dropout:
		x = Dropout(dropout)(x)
	
	x = Dense(n_channels * 32, kernel_regularizer = regularizer, activation = "relu")(x)

	if dropout:
		x = Dropout(dropout)(x)
	output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

	model = Model(input_, output)
	return model



if __name__ == "__main__":

	# Define the constants
	DATASET_PATH = "./Gender/"
	BATCH_SIZE = 32
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

				print(type(data))

				retrival_connector.send(data)
				metadata = retrival_connector.recv(1024)
				metadata = json.loads(metadata.decode("utf-8"))

				if "status" not in metadata:
					raise KeyError("Missing status in the metadata")
				else:
					if metadata["status"] == True:
						embedding = retrival_connector.recv(1024)
					else:
						embedding = None

				print(metadata)
				camera.send(json.dumps({"age": metadata["age"], "gender": metadata["gender"]}).encode("UTF-8"))

