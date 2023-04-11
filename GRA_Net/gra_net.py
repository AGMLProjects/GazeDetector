from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Activation, MaxPool2D, Add, Multiply, Lambda, Layer
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas, os
from PIL import Image
from keras.layers import add
from sklearn.model_selection import train_test_split


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
    
def residual_block(input, input_channels = None, output_channels = None, kernel_size = (3, 3), stride = 1):

    if output_channels is None:
        output_channels = input.get_shape()[-1]

    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1), strides = (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, strides = strides, padding = 'same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding = "same")(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), strides = strides, padding = 'same')(input)

    x = Add()([x, input])
    return x

def attention_block(input, input_channels = None, output_channels = None, encoder_depth = 1):
    
    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1]

    if output_channels is None:
        output_channels = input_channels

    # First residual block
    for i in range(p):
        input = residual_block(input)

    # Trunk Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch

    # Encoder
    # First downsampling
    output_soft_mask = MaxPool2D(padding = 'same')(input)   # 32x32

    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        # Skip connections
        output_skip_connections = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connections)
        print("Skip connection shape: ", output_skip_connections.get_shape())

        # Downsampling
        output_soft_mask = MaxPool2D(padding = 'same')(output_soft_mask)

        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

    # Decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        # Upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
            output_soft_mask = UpSampling2D()(output_soft_mask)

            # Skip connections
            output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    # Last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)

    # Output
    output_soft_mask = Conv2D(output_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(output_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention
    output = GatedLayer()(output_soft_mask)
    output = Multiply()([output, output_trunk])

    # Last residual block
    for i in range(p):
        output = residual_block(output)

    return output

if __name__ == "__main__":

    # Import the dataset in the form of a pandas dataframe
    age_list, gender_list, race_list = list(), list(), list()
    data_list = list()

    for filename in os.listdir("D:/UTKface_inthewild/part1/"):
        args = filename.split("_")

        if len(args) < 4:
            age = int(args[0])
            gender = int(args[1])
            race = 4
        else:
            age = int(args[0])
            gender = int(args[1])
            race = int(args[2])

        age_list.append(age)
        gender_list.append(gender)
        race_list.append(race)
        data_list.append("D:/UTKface_inthewild/part1/" + filename)


    dataframe = pandas.DataFrame({
        "filename": data_list,
        "label": gender_list,
    })

    dataframe["label"] = dataframe["label"].map({0: "male", 1: "female"})

    # Set up the training and testing data generators
    train_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    # Split the dataframe into train and test datasets
    train_df, test_df = train_test_split(dataframe, test_size = 0.2)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col = "filename",
        y_col = "label",
        target_size = (32, 32),
        batch_size = 32,
        class_mode = 'binary')
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col = "filename",
        y_col = "label",
        target_size = (32, 32),
        batch_size = 32,
        class_mode = 'binary')
    
    # Set up the Gated Residual Attention Network architecture
    inputs = Input(shape = (32, 32, 3))

    # Initial Convolution and MaxPooling
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPool2D(pool_size = (2, 2))(x)

    # First attention block with encoder depth of 3
    x = attention_block(x, encoder_depth = 3)
    x = residual_block(x)

    # Second attention block with encoder depth of 2
    x = attention_block(x, encoder_depth = 2)
    x = residual_block(x)

    # Third attention block with encoder depth of 1
    x = attention_block(x, encoder_depth = 1)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)

    # Final Convolution and MaxPooling
    x = MaxPool2D(pool_size = (2, 2))(x)

    # Dense layers
    x = Dense(1, activation = 'relu')(x)

    # Compile the model
    model = Model(inputs = inputs, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])

    # Train the model
    checkpoint = ModelCheckpoint('model.h5', monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 10,
        validation_data = test_generator,
        validation_steps = 50,
        callbacks = [checkpoint])
    
    # Evaluate the model
    model.load_weights('model.h5')
    scores = model.evaluate_generator(test_generator, steps = 50)

    print("Accuracy: ", scores[1])
    print("Loss: ", scores[0])
    
    