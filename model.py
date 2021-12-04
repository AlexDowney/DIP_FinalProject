import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from build import build_vae_encoder, build_vae_decoder

def build_vae_model():
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
    
    # The input to the model will be the image fed to the encoder.
    vae_input = vae_encoder_input
    
    # Output will be the output of the decoder. The term - decoder(encoder_output) 
    # combines the model by passing the encoder output to the input of the decoder.
    vae_output = vae_decoder(vae_encoder_output)
    
    # Input to the combined model will be the input to the encoder.
    # Output of the combined model will be the output of the decoder.
    vae_model = Model(vae_input, vae_output)
    
    return vae_model, vae_encoder_input, vae_encoder_output, mean_mu, log_var, vae_shape_before_flattening, vae_encoder, vae_decoder_input, vae_decoder_output, vae_decoder

if __name__ == "__main__":
    vae_model, vae_encoder_input, vae_encoder_output, mean_mu, log_var, vae_shape_before_flattening, vae_encoder, vae_decoder_input, vae_decoder_output, vae_decoder = build_vae_model()
    vae_model.summary()

