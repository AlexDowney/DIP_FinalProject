import os
from glob import glob
from PIL import Image
import tensorflow as tf
import numpy as np
# import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from model import build_mlp_model

import sys
sys.path.insert(1, '../')

from VAE_Scripts import *


def dataset_gen():
    x_train = np.load('x_train.npy', allow_pickle=True)
    y_train = np.load('y_train.npy', allow_pickle=True)
    
    for i in range(len(x_train)):
        x = np.array(x_train[i])
        y = np.array(y_train[i])
        yield x,y


def x_gen():
    x_train = np.load('x_train.npy', allow_pickle=True)
    
    for i in range(len(x_train)):
        x = np.array(x_train[i])
        yield x


def y_gen():
    y_train = np.load('y_train.npy', allow_pickle=True)
    
    for i in range(len(y_train)):
        y = np.array(y_train[i])
        yield y


def train_model():
    WEIGHTS_FOLDER = './weights/'
    DATA_FOLDER = '../data/'
    
    INPUT_DIM = (128,128,3) # Image dimension
    BATCH_SIZE = 512
    
    filenames = np.array(glob(os.path.join(DATA_FOLDER, 'img_align_celeba/*/*.jpg')))
    NUM_IMAGES = len(filenames)
    
    attr_file = open(os.path.join(DATA_FOLDER, 'list_attr_celeba.csv'))
    file_contents = np.loadtxt(attr_file,delimiter=",",dtype=object)
    attr_file.close()
    
    attr_labels = file_contents[:1, 1:]
    attr_matrix = file_contents[1:, 1:]
    image_names = file_contents[1:, :1]
    
    NUM_IMAGES = len(image_names.flatten())
    
    
    # x_train = attr_matrix.flatten().reshape((1,202599,40,1))
    
    # vae_model, vae_encoder, vae_decoder = train.train_model()
    
    # y_train = list(range(len(image_names.flatten())))
    # for i in range(len(image_names)):
        # name = image_names[i]
        
        # image = np.asarray(Image.open('../data/img_align_celeba/img_align_celeba/'+ name[0]).resize((128,128), Image.ANTIALIAS)).reshape((1,128,128,3))/255
        
        # y_train[i] = vae_encoder.predict(image)
        # if (i%1000 == 0):
            # print(i)
    # y_train = np.array(y_train).flatten().reshape((1,202599,200,1))
    
    
    
    # x_train.dump('x_train.npy')
    # y_train.dump('y_train.npy')
    
    
    
    
    mlp_model = build_mlp_model()
    
    LEARNING_RATE = 0.000005
    N_EPOCHS = 200
    
    def r_loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred), axis = [1])
    
    def cos_loss(y_true, y_pred):
        return tf.keras.losses.cosine_similarity(y_true, y_pred)
    
    # mse = tf.keras.losses.MeanSquaredError()
    
    adam_optimizer = Adam(learning_rate = LEARNING_RATE)
    
    mlp_model.compile(optimizer=adam_optimizer, loss = r_loss, metrics = [cos_loss, r_loss])
    
    checkpoint_mlp = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'MLP/weights.h5'), save_weights_only = True, verbose=1)

    checkpoint_mlp_best = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'MLP/best_weights.h5'), save_weights_only = True, verbose=1, mode='max')

    tensorboard_callback = TensorBoard(log_dir='./logs/'+'model_training_results')
    
    
    x_train = np.load('x_train.npy', allow_pickle=True).squeeze()
    y_train = np.load('y_train.npy', allow_pickle=True).squeeze()
    
    print(x_train.shape)
    print(y_train.shape)
    
    # dataset = dataset_gen()
    
    
    mlp_model.fit(
        x_train, 
        y_train, 
        shuffle=True, 
        epochs = N_EPOCHS, 
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_mlp, checkpoint_mlp_best, tensorboard_callback])


if __name__ == "__main__":
    train_model()







