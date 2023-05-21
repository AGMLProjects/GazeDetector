from keras.layers import AveragePooling2D, Layer, Dropout, BatchNormalization, Conv2D, UpSampling2D, Activation, MaxPooling2D, Add, Multiply, Input, Dense, Flatten
from keras.models import Model
from keras.regularizers import l2

import tensorflow as tf

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

def GenderNetwork(shape = (224, 224, 3), n_channels = 64, n_classes = 2, dropout = 0, regularization = 0.01):

    # Define the regularization factor
    regularizer = l2(regularization)

    # Perform preliminar normalization, reshaping and pooling before feeding the attention network
    input_ = Input(shape = shape)
    x = Conv2D(n_channels, (7, 7), strides = (2, 2), padding = "same")(input_)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same")(x)

    # Feed the GRA_Net
    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    # Final pooling and softmax steps
    pool_size = (x.get_shape()[1], x.get_shape()[2])
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

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
        train = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            validation_split = 0.2,
            subset = "training",
            seed = 123,
            image_size = image_size,
            batch_size = batch_size
        )

        validation = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            validation_split = 0.2,
            subset = "validation",
            seed = 123,
            image_size = image_size,
            batch_size = batch_size
        )

        # Print the dataset classes
        class_names = train.class_names
        print(class_names)

        # Define the model
        model = GenderNetwork(shape = (224, 224, 3), n_channels = 64, n_classes = 2, dropout = 0, regularization = 0.01)
        model.summary()

    # Compile the model
    model.compile(
        optimizer = "adam",
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
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

if __name__ == "__main__":

    # Define the constants
    IMAGE_SIZE = (224, 224)
    DATASET_PATH = "./Gender/"
    BATCH_SIZE = 32
    EPOCHS = 10



