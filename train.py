import os
from glob import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam# adam_v2
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from model import build_vae_model


def train_model():
    WEIGHTS_FOLDER = './weights/'
    DATA_FOLDER = './data/img_align_celeba/'


    filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
    NUM_IMAGES = len(filenames)
    print("Total number of images : " + str(NUM_IMAGES))
    # prints : Total number of images : 202599


    INPUT_DIM = (128,128,3) # Image dimension
    BATCH_SIZE = 512


    data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(
        DATA_FOLDER, 
        target_size = INPUT_DIM[:2],
        batch_size = BATCH_SIZE,
        shuffle = True,
        class_mode = 'input',
        subset = 'training'
    )

    LEARNING_RATE = 0.0005
    N_EPOCHS = 200
    LOSS_FACTOR = 10000

    def r_loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

    def kl_loss(y_true, y_pred):
        kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis = 1)
        return kl_loss

    def total_loss(y_true, y_pred):
        return LOSS_FACTOR*r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

    adam_optimizer = Adam(learning_rate = LEARNING_RATE) # adam_v2.Adam(lr = LEARNING_RATE)

    vae_model, vae_encoder_input, vae_encoder_output, mean_mu, log_var, vae_shape_before_flattening, vae_encoder, vae_decoder_input, vae_decoder_output, vae_decoder = build_vae_model()

    vae_model.compile(optimizer=adam_optimizer, loss = total_loss, metrics = [r_loss, kl_loss])

    checkpoint_vae = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'VAE/weights.h5'), save_weights_only = True, verbose=1)

    checkpoint_vae_best = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'VAE/best_weights.h5'), save_weights_only = True, verbose=1, mode='max')

    tensorboard_callback = TensorBoard(log_dir='./logs/'+'model_training_results')

    #vae_model.fit   (data_flow, 
    #                shuffle=True, 
    #                epochs = N_EPOCHS, 
    #                initial_epoch = 0, 
    #                steps_per_epoch=NUM_IMAGES / BATCH_SIZE,
    #                callbacks=[checkpoint_vae, checkpoint_vae_best, tensorboard_callback])

    vae_model.load_weights(os.path.join(WEIGHTS_FOLDER, 'VAE/best_weights.h5'))
    return vae_model, vae_encoder, vae_decoder


if __name__ == "__main__":
    vae_model, vae_encoder, vae_decoder = train_model()
    vae_model.summary()

