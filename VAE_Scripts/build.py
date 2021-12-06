import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# ENCODER
def build_vae_encoder(input_dim, output_dim, conv_filters, conv_kernel_size, conv_strides, use_batch_norm = False, use_dropout = False):

    # Clear tensorflow session to reset layer index numbers to 0 for LeakyRelu, 
    # BatchNormalization and Dropout.
    # Otherwise, the names of above mentioned layers in the model 
    # would be inconsistent
    global K
    K.clear_session()

    # Number of Conv layers
    n_layers = len(conv_filters)

    # Define model input
    encoder_input = Input(shape = input_dim, name = 'encoder_input')
    x = encoder_input

    # Add convolutional layers
    for i in range(n_layers):
        x = Conv2D(filters = conv_filters[i], 
                   kernel_size = conv_kernel_size[i],
                   strides = conv_strides[i], 
                   padding = 'same',
                   name = 'encoder_conv_' + str(i)
                   )(x)
        if use_batch_norm:
            x = BathcNormalization()(x)

        x = LeakyReLU()(x)

        if use_dropout:
            x = Dropout(rate=0.25)(x)

    # Required for reshaping latent vector while building Decoder
    shape_before_flattening = K.int_shape(x)[1:] 

    x = Flatten()(x)

    mean_mu = Dense(output_dim, name = 'mu')(x)
    log_var = Dense(output_dim, name = 'log_var')(x)

    # Defining a function for sampling
    def sampling(args):
        mean_mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.) 
        return mean_mu + K.exp(log_var/2)*epsilon   

    # Using a Keras Lambda Layer to include the sampling function as a layer 
    # in the model
    encoder_output = Lambda(sampling, name='encoder_output')([mean_mu, log_var])

    return encoder_input, encoder_output, mean_mu, log_var, shape_before_flattening, Model(encoder_input, encoder_output)


# Decoder
def build_vae_decoder(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, conv_strides):

    # Number of Conv layers
    n_layers = len(conv_filters)

    # Define model input
    decoder_input = Input(shape = (input_dim,) , name = 'decoder_input')

    # To get an exact mirror image of the encoder
    x = Dense(np.prod(shape_before_flattening))(decoder_input)
    x = Reshape(shape_before_flattening)(x)

    # Add convolutional layers
    for i in range(n_layers):
        x = Conv2DTranspose(filters = conv_filters[i], 
                    kernel_size = conv_kernel_size[i],
                    strides = conv_strides[i], 
                    padding = 'same',
                    name = 'decoder_conv_' + str(i)
                    )(x)
      
      # Adding a sigmoid layer at the end to restrict the outputs 
      # between 0 and 1
        if i < n_layers - 1:
            x = LeakyReLU()(x)
        else:
            x = Activation('sigmoid')(x)

    # Define model output
    decoder_output = x

    return decoder_input, decoder_output, Model(decoder_input, decoder_output)


if __name__ == "__main__":
    INPUT_DIM = (128,128,3) # Image dimension
    Z_DIM = 200 # Dimension of the latent vector (z)
    
    vae_encoder_input, vae_encoder_output, mean_mu, log_var, vae_shape_before_flattening, vae_encoder = build_vae_encoder(
        input_dim = INPUT_DIM,
        output_dim = Z_DIM, 
        conv_filters = [32, 64, 64, 64],
        conv_kernel_size = [3,3,3,3],
        conv_strides = [2,2,2,2]
    )
    
    vae_decoder_input, vae_decoder_output, vae_decoder = build_vae_decoder(
        input_dim = Z_DIM,
        shape_before_flattening = vae_shape_before_flattening,
        conv_filters = [64,64,32,3],
        conv_kernel_size = [3,3,3,3],
        conv_strides = [2,2,2,2]
    )
    
    vae_encoder.summary();
    vae_decoder.summary();

