from keras.layers import AveragePooling2D, Layer, Dropout, BatchNormalization, Conv2D, UpSampling2D, Activation, MaxPooling2D, Add, Multiply, Input, Dense, Flatten
from keras.models import Model
from keras.regularizers import l2

from tensorflow import keras
import tensorflow as tf

from PIL import Image
import numpy as np
import pandas as pd
import os, re, pathlib

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
    output = Multiply()([output, output_trunk])

    # Last residual block
    for i in range(p):
        output = residual_block(input = output)

    return output

def GenderNetwork(shape = (32, 32, 3), n_channels = 64, n_classes = 2, dropout = 0, regularization = 0.001):

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
    
    x = residual_block(x, output_channels=n_channels * 32, stride=2)    # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    # Final pooling and softmax steps
    pool_size = (x.get_shape()[1], x.get_shape()[2])
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='relu')(x)

    model = Model(input_, output)
    return model

def train_model(image_size: tuple, dataset_path: str, batch_size: int = 32, epochs: int = 10):
    if type(image_size) != tuple or type(dataset_path) != str or type(batch_size) != int or type(epochs) != int:
        raise TypeError("Wrong type of input parameters")

    # Define the distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    # Define the GPU device
    with strategy.scope():
        # Load the dataset
        train, validation = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            labels='inferred',
            label_mode = "binary",
            validation_split = 0.2,
            subset = "both",
            seed = 123,
            image_size = image_size,
            batch_size = batch_size
        )

        # Print the dataset classes
        class_names = train.class_names
        print(class_names)

        # Define the model
        model = GenderNetwork(shape = (32, 32, 3), n_channels = 64, n_classes = 2, dropout = 0, regularization = 0.001)
        model.summary()

        optimizer = tf.keras.optimizers.Nadam()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
        metrics = [tf.keras.metrics.BinaryAccuracy()]

    # Compile the model
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrics
    )

    # Train the model
    model.fit(
        train,
        validation_data = validation,
        epochs = epochs
    )

    # Print the accuracy
    print("Accuracy: {}".format(model.evaluate(validation)[1]))

    # Save the model
    model.save("GenderNetwork.h5")


def get_label(file_path):

    # Convert the path to a list of metadata
    metadata = tf.strings.split(file_path, "_")
    if len(metadata) < 4 or int(metadata[1]) == 0:
        return 0
    elif int(metadata[1]) == 1:
        return 1
    else:
        return 1

def decode_image(img, image_size):

    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)

    # Resize the image to the desired size
    return tf.image.resize(img, image_size)

def process_path(file_path):
    label = get_label(file_path = file_path)

    img = tf.io.read_file(file_path)
    img = decode_image(img = img, image_size = [32, 32])
    return img, label

def configure_dataset(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size = 1000)
    ds = ds.batch(32)
    ds = ds.prefetch(buffer_size = tf.data.AUTOTUNE)
    
    return ds

if __name__ == "__main__":

    # Define the constants
    IMAGE_SIZE = (32, 32)
    DATASET_PATH = "./Gender/"
    BATCH_SIZE = 32
    EPOCHS = 10
    CLASS_NAME = ["female", "male"]
    DATA_REGEX = r"([0-9]+)[\_]+([0-9]+)[\_]+([0-9]+)[\_]+([0-9]+)\.jpg"

    # Define the distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    # Define the GPU device
    with strategy.scope():
        
        # Define the Dataset load pipeline
        data_directory = pathlib.Path(DATASET_PATH)
        image_count = len(list(data_directory.glob("*.jpg")))
        print(f"Dataset size: {image_count}")
        
        # Load the dataset and shuffle the images
        list_ds = tf.data.Dataset.list_files(str(data_directory/"*"), shuffle = False)
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration = False)

        # Define the class names
        class_names = np.array(sorted(["male", "female"]))

        # Split the dataset
        val_size = int(image_count * 0.2)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)

        train_ds = train_ds.map(process_path, num_parallel_calls = tf.data.AUTOTUNE)
        val_ds = val_ds.map(process_path, num_parallel_calls = tf.data.AUTOTUNE)

        train_ds = configure_dataset(train_ds)
        val_ds = configure_dataset(val_ds)

        # Define the model
        model = GenderNetwork(shape = (32, 32, 3), n_channels = 64, n_classes = 2, dropout = 0, regularization = 0.001)
        model.summary()

        optimizer = tf.keras.optimizers.Nadam()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
        metrics = [tf.keras.metrics.BinaryAccuracy()]

    # Compile the model
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrics
    )

    # Train the model
    model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = EPOCHS
    )

    # Print the accuracy
    print("Accuracy: {}".format(model.evaluate(val_ds)[1]))

    # Save the model
    model.save("GenderNetwork.h5")