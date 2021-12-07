import os
from glob import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import sys
from PIL import Image
sys.path.insert(1, '../')
from VAE_Scripts import *
from MLP_Scripts import *


def vec_comb(img_vec, attr_vec):
    ratio = .3
    return (ratio*attr_vec) + ((1-ratio)*img_vec)
    # vec_mask = attr_vec - img_vec
    # scaled_mask = .5 * vec_mask
    # return scaled_mask + img_vec
    
    # vec_mask = (img_vec+img_vec)/2
    # vec_mask[0][0] += .1
    # vec_mask[0][12] -= .1
    # vec_mask[0][33] += .1
    # return vec_mask


def impose_attr(image, attr, vae_encoder, vae_decoder, mlp_model):
    
    img_vec = vae_encoder.predict(image)
    attr_vec = mlp_model.predict(attr)
    
    new_vec = vec_comb(img_vec, attr_vec)
    
    new_img = vae_decoder.predict(new_vec)
    
    rec_img = vae_model.predict(image)
    
    plt.figure()
    plt.imshow(image.squeeze())
    
    plt.figure()
    plt.imshow(rec_img.squeeze())
    
    plt.figure()
    plt.imshow(new_img.squeeze())
    
    plt.show()


if __name__ == "__main__":
    WEIGHTS_FOLDER = './weights/'
    DATA_FOLDER = '../data/img_align_celeba/'

    vae_model, vae_encoder, vae_decoder = train_model()
    
    mlp_model = build_mlp_model()
    mlp_model.load_weights(os.path.join(WEIGHTS_FOLDER, 'MLP/best_weights.h5'))
    
    
    image = Image.open(DATA_FOLDER+'img_align_celeba/'+'000001.jpg')
    # image = Image.open('../test_images/test_images/test_image.jpg')
    image = image.resize((128,128), Image.ANTIALIAS)
    image = np.asarray(image).reshape((1,128,128,3))
    image = image/255
    
    
    # print(image)
    
    # INPUT_DIM = (128,128,3) # Image dimension
    # data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(
        # "../data/img_align_celeba/", 
        # target_size = INPUT_DIM[:2],
        # batch_size = 1,
        # shuffle = True,
        # class_mode = 'input',
        # subset = 'training'
    # )
    
    # example_batch = next(data_flow)
    # example_batch = example_batch[0]
    # example_images = example_batch[:1]
    # image = example_images
    # print(image)
    
    attr = np.array([[-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 15, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1]]).reshape((1,40))
    # attr = np.array([[1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1]]).reshape((1,40))

    
    attr_labels = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
    
    impose_attr(image, attr, vae_encoder, vae_decoder, mlp_model)

