import os
from glob import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from train import train_model


def plot_compare_vae(vae_model, images=None):
    if images is None:
        example_batch = next(data_flow)
        example_batch = example_batch[0]
        images = example_batch[:10]
    
    n_to_show = images.shape[0]
    reconst_images = vae_model.predict(images)
    
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    for i in range(n_to_show):
        img = images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')
        sub.imshow(img)
    
    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
        sub.axis('off')
        sub.imshow(img)
    plt.plot()


def vae_generate_images(vae_decoder, n_to_show=10, Z_DIM=200):
    reconst_images = vae_decoder.predict(np.random.normal(0,1,size=(n_to_show,Z_DIM)))
    
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    for i in range(n_to_show):
            img = reconst_images[i].squeeze()
            sub = fig.add_subplot(2, n_to_show, i+1)
            sub.axis('off')
            sub.imshow(img)
    plt.plot()


def vae_generate_mod_images(vae_encoder, vae_decoder, image, Z_DIM=200):
    import copy
    n_to_show = image.shape[0]
    face_vec = vae_encoder.predict(image)
    noise_vec = np.random.normal(0,1,size=(n_to_show,Z_DIM))
    
    # print(face_vec)
    # print(noise_vec)
    
    final_vec = copy.copy(face_vec);
    for i in range(len(final_vec)):
        final_vec[i] = (.8*face_vec[i] + .2*noise_vec[i])
    
    reconst_images = vae_decoder.predict(face_vec)
    
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    for i in range(n_to_show):
        img = image[i].squeeze()
        sub = fig.add_subplot(3, n_to_show, i+1)
        sub.axis('off')
        sub.imshow(img)
    
    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(3, n_to_show, i+n_to_show+1)
        sub.axis('off')
        sub.imshow(img)
    
    reconst_images = vae_decoder.predict(final_vec)
    
    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(3, n_to_show, i+n_to_show+2)
        sub.axis('off')
        sub.imshow(img)
    
    plt.plot()


if __name__ == "__main__":
    vae_model, vae_encoder, vae_decoder = train_model()
    
    WEIGHTS_FOLDER = './weights/'
    DATA_FOLDER = '../data/img_align_celeba/'

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

    example_batch = next(data_flow)
    example_batch = example_batch[0]
    example_images = example_batch[:10]
    
    plot_compare_vae(vae_model, images = example_images)
    
    vae_generate_images(vae_decoder, n_to_show=20)
    
    data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(
        "../test_images/", 
        target_size = INPUT_DIM[:2],
        batch_size = 1,
        shuffle = True,
        class_mode = 'input',
        subset = 'training'
    )
    
    example_batch = next(data_flow)
    example_batch = example_batch[0]
    example_images = example_batch[:1]
    
    # plot_compare_vae(vae_model, images = example_images)
    # vae_generate_mod_images(vae_encoder, vae_decoder, example_images)
    
    
    plt.show()

