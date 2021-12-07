import os
from glob import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.insert(1, '../')
from VAE_Scripts import *
from MLP_Scripts import *


def vec_comb(img_vec, attr_vec, ratio):
    return (ratio*attr_vec) + ((1-ratio)*img_vec)


def get_attr_vec(attr):
    mlp_model = build_mlp_model()
    mlp_model.load_weights(os.path.join('weights/MLP/best_weights.h5'))
    return mlp_model.predict(attr)


if __name__ == "__main__":
    WEIGHTS_FOLDER = '../MLP_Scripts/weights/'
    
    vae_model, vae_encoder, vae_decoder = train_model()
    
    
    image1 = Image.open('000001.jpg')
    image1 = image1.resize((128,128), Image.ANTIALIAS)
    image1 = np.asarray(image1).reshape((1,128,128,3))
    image1 = image1/255
    
    image2 = Image.open('000007.jpg')
    image2 = image2.resize((128,128), Image.ANTIALIAS)
    image2 = np.asarray(image2).reshape((1,128,128,3))
    image2 = image2/255
    
    img_vec_1 = vae_encoder.predict(image1)
    img_vec_2 = vae_encoder.predict(image2)
    
    
    plt.figure()
    plt.imshow(image1.squeeze())
    
    image = vae_decoder.predict(img_vec_1)
    plt.figure()
    plt.imshow(image.squeeze())
    
    # Bald .7 10
    attr = np.array([[-1, 1, 1, -1, 10, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1]], dtype=object).reshape((1,40))
    attr_vec = get_attr_vec(attr)
    image = vae_decoder.predict(vec_comb(img_vec_1, attr_vec, .7))
    plt.figure()
    plt.imshow(image.squeeze())
    
    # DarkHair .4 8
    attr = np.array([[-1, 1, 1, -1, -1, -1, -1, -1, 8, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1]]).reshape((1,40))
    attr_vec = get_attr_vec(attr)
    image = vae_decoder.predict(vec_comb(img_vec_1, attr_vec, .4))
    plt.figure()
    plt.imshow(image.squeeze())
    
    # SunGlasses .3 15
    attr = np.array([[-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 15, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1]]).reshape((1,40))
    attr_vec = get_attr_vec(attr)
    image = vae_decoder.predict(vec_comb(img_vec_1, attr_vec, .3))
    plt.figure()
    plt.imshow(image.squeeze())
    
    # Hat .5 10
    attr = np.array([[-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 10, 1, -1, -1, 1]]).reshape((1,40))
    attr_vec = get_attr_vec(attr)
    image = vae_decoder.predict(vec_comb(img_vec_1, attr_vec, .5))
    plt.figure()
    plt.imshow(image.squeeze())
    
    
    plt.figure()
    plt.imshow(image2.squeeze())
    
    image = vae_decoder.predict(img_vec_2)
    plt.figure()
    plt.imshow(image.squeeze())
    
    # Woman .5 -10
    attr = np.array([[1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -10, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1]]).reshape((1,40))
    attr_vec = get_attr_vec(attr)
    image = vae_decoder.predict(vec_comb(img_vec_2, attr_vec, .5))
    plt.figure()
    plt.imshow(image.squeeze())
    
    # Man .5 10
    attr = np.array([[1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1]]).reshape((1,40))
    attr_vec = get_attr_vec(attr)
    image = vae_decoder.predict(vec_comb(img_vec_2, attr_vec, .5))
    plt.figure()
    plt.imshow(image.squeeze())
    
    # Bald .7 10
    attr = np.array([[1, -1, 1, 1, 10, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1]]).reshape((1,40))
    attr_vec = get_attr_vec(attr)
    image = vae_decoder.predict(vec_comb(img_vec_2, attr_vec, .7))
    plt.figure()
    plt.imshow(image.squeeze())
    
    # Smile .4 5
    attr = np.array([[1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 5, 1, -1, -1, -1, -1, -1, -1, 1]]).reshape((1,40))
    attr_vec = get_attr_vec(attr)
    image = vae_decoder.predict(vec_comb(img_vec_2, attr_vec, .4))
    plt.figure()
    plt.imshow(image.squeeze())
    
    plt.show()

