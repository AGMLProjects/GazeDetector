from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Activation, MaxPooling2D, Add, Multiply, Lambda
from keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer

import numpy as np

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

