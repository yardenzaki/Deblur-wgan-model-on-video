## README:
"""
This Script applies a trained Deblurring GAN model on a "blurry input (video)" to demonstrate the improvement in sharpness

Inputs:
* START_VID_FROM_TIME = 3  # start the video input from from .... [s]
* STOP_AT_TIME = 8.5 # end capturing the video at time.... [s]
* wait_key = 300  # Time in ms between frames
* save_frames = True # flag to decide if to save the output frames or to just show
* IMAGE_DIM_PARAM = 128 # GAN model input shape - in this case the GAN receives 128*128*1 images
* BATCH_SIZE = 20  # Batch size

* Blur_Valid_DS =os.path.join("C:/Users/E010236/Downloads","Blur_Valid_DS.npy") # Blurred Validation Data Set
* Sharp_Valid_DS =os.path.join("C:/Users/E010236/Downloads","Sharp_Valid_DS.npy") # Sharp Validation Data Set
* hdf_path = os.path.join("C:/Users/E010236/Downloads","my_best_model.epoch11-loss47.87.hdf5") # Path to HDF5 file of trained model **weights only**

steps:
1. Run the script and pick any video.
2. select ROI to crop the video to the wanted key region.
3. Apply selection by ENTER/SPACE key and Execute by ESC key.

Output:
* Example Blur / Deblur images for the first frame of the video
* Output Video that contains the deblurring results.

Notes:
* GAN input is NORMALIZED image.
* due to padding = "same" , reconstructed patched image size is bigger than the original and therefore we use CenterCrop at the end of the process
* GAN's generator output is mapped to 0-255 by the function "interval_mapping(image, from_min, from_max, to_min, to_max)"

@Author: Yarden Zaki
@Date: 07/01/2022
@Version: 1.0
@Links: https://github.com/yardenzaki
@License: MIT
"""

import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
# import datetime
import imutils
from imutils import paths
import argparse
import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers, preprocessing
import matplotlib.pyplot as plt

###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
START_VID_FROM_TIME = 8  # start the video input from from .... [s]
STOP_AT_TIME = 8.5  # end capturing the video at time.... [s]
wait_key = 300  # Time in ms between frames
save_frames = True
IMAGE_DIM_PARAM = 128  # Wanted images dimension
BATCH_SIZE = 20  # 5 # Batch size

Blur_Valid_DS = os.path.join("C:/Users/E010236/Downloads", "Blur_Valid_DS.npy")
Sharp_Valid_DS = os.path.join("C:/Users/E010236/Downloads", "Sharp_Valid_DS.npy")
hdf_path = os.path.join("C:/Users/E010236/Downloads", "my_best_model.epoch11-loss47.87.hdf5")

Blur_Valid_DS = np.load(Blur_Valid_DS)  # load
Sharp_Valid_DS = np.load(Sharp_Valid_DS)  # load

Blur_Train_DS = Blur_Valid_DS
Sharp_Train_DS = Sharp_Valid_DS


###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
def ResBlock(X, kernel, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=0)

    # defining name basis
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1 = filters
    F2 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = layers.Conv2D(filters=F1, kernel_size=(kernel, kernel), strides=(1, 1), padding='same',
                      name=conv_name_base + '2a', kernel_initializer=initializer)(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.LeakyReLU()(X)

    # Second component of main path (≈3 lines)
    X = layers.Conv2D(filters=F2, kernel_size=(kernel, kernel), strides=(1, 1), padding='same',
                      name=conv_name_base + '2b', kernel_initializer=initializer)(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.LeakyReLU()(X)

    # # Third component of main path (≈2 lines)
    # X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'same', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])

    return X


def OrangeBlock(X, kernel, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=0)

    # defining name basis
    avg_pool_name_base = 'avg_pool' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    conv_name_base = 'conv' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1 = filters

    # Shortcut path
    X_shortcut = layers.Conv2D(filters=F1, kernel_size=(kernel, kernel), strides=(2, 2), padding='same',
                               name=conv_name_base + '2a', kernel_initializer=initializer)(X)
    X_shortcut = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X_shortcut)
    X_shortcut = layers.LeakyReLU()(X_shortcut)

    # Avg pooling path
    X = layers.AveragePooling2D(pool_size=(2, 2))(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])

    return X


def GenericBlock(X, kernel, stride, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=0)

    # defining name basis
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1 = filters

    # First component of main path
    X = layers.Conv2D(filters=F1, kernel_size=(kernel, kernel), strides=(stride, stride), padding='same',
                      name=conv_name_base + '1', kernel_initializer=initializer)(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X)
    X = layers.LeakyReLU()(X)
    return X


def GreenBlock(X, kernel, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=0)

    # defining name basis
    conv_name_base = 'conv' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1 = filters

    # First component of main path
    X = layers.Conv2D(filters=F1, kernel_size=(kernel, kernel), strides=(1, 1), padding='same',
                      name=conv_name_base + '1', kernel_initializer=initializer)(X)
    X = tf.keras.activations.tanh(X)
    return X


def YellowBlock(X, kernel, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=0)

    # defining name basis
    conv_name_base = 'DEconv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1 = filters

    # First component of main path
    X = layers.Conv2DTranspose(filters=F1, kernel_size=(kernel, kernel), strides=(2, 2), padding='same',
                               name=conv_name_base + '1', kernel_initializer=initializer)(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X)
    X = layers.LeakyReLU()(X)
    return X


def BiSkip(input_dim):
    """
    BiSkip Generator:
    input: blur image (X)
    output: sharp reconstructed image (X_hat)


    Definitions:
    X_shortcut_R_{int} ; saved output for skip connection from RedBlocks
    X_shortcut_T_{int} ; saved output for skip connection from TurquiseBlocks
    """
    X_input = layers.Input(shape=(input_dim, input_dim, 1))

    # save shortcut for addition
    X_input_shortcut = X_input

    # stage - 1
    stage = 1
    X = GenericBlock(X_input, kernel=7, stride=1, filters=32, stage=stage, block='a')  # NavyBlock
    X_Navy_1 = X
    X = GenericBlock(X, kernel=1, stride=1, filters=16, stage=stage, block='b')  # RedBlock
    X_shortcut_R_1 = X  # Save for skip connection

    # stage - 2
    stage = 2
    X = GenericBlock(X_Navy_1, kernel=3, stride=1, filters=32, stage=stage, block='a')  # GrayBlock

    # stage - 3
    stage = 3
    X = ResBlock(X, kernel=3, filters=32, stage=stage, block='a')  # ResBlock
    X = ResBlock(X, kernel=3, filters=32, stage=stage, block='b')  # ResBlock
    X = ResBlock(X, kernel=3, filters=32, stage=stage, block='c')  # ResBlock
    X_Green_3 = X
    X = GenericBlock(X_Green_3, kernel=3, stride=2, filters=16, stage=stage, block='d')  # TurquiseBlock
    X_shortcut_T_3 = X  # Save for skip connection

    # stage - 4
    stage = 4
    X = OrangeBlock(X_Green_3, kernel=5, filters=32, stage=stage, block='a')  # OrangeBlock
    X_Orange_4 = X
    X = GenericBlock(X, kernel=1, stride=1, filters=32, stage=stage, block='b')  # RedBlock
    X_shortcut_R_4 = X  # Save for skip connection

    #################-------------------------------------------------------------------

    # stage - 5
    stage = 5
    X = GenericBlock(X_Orange_4, kernel=3, stride=1, filters=64, stage=stage, block='a')  # GrayBlock

    # stage - 6
    stage = 6
    X = ResBlock(X, kernel=3, filters=64, stage=stage, block='a')  # ResBlock
    X = ResBlock(X, kernel=3, filters=64, stage=stage, block='b')  # ResBlock
    X = ResBlock(X, kernel=3, filters=64, stage=stage, block='c')  # ResBlock
    X_Green_6 = X
    X = GenericBlock(X_Green_6, kernel=3, stride=2, filters=32, stage=stage, block='d')  # TurquiseBlock
    X_shortcut_T_6 = X  # Save for skip connection

    # stage - 7
    stage = 7
    X = OrangeBlock(X_Green_6, kernel=5, filters=64, stage=stage, block='a')  # OrangeBlock
    X_Orange_7 = X
    X = GenericBlock(X, kernel=1, stride=1, filters=64, stage=stage, block='b')  # RedBlock
    X_shortcut_R_7 = X  # Save for skip connection

    #################-------------------------------------------------------------------

    # stage - 8
    stage = 8
    X = GenericBlock(X_Orange_7, kernel=3, stride=1, filters=128, stage=stage, block='a')  # GrayBlock

    # stage - 9
    stage = 9
    X = ResBlock(X, kernel=3, filters=128, stage=stage, block='a')  # ResBlock
    X = ResBlock(X, kernel=3, filters=128, stage=stage, block='b')  # ResBlock
    X = ResBlock(X, kernel=3, filters=128, stage=stage, block='c')  # ResBlock
    X_Green_9 = X
    X = GenericBlock(X_Green_9, kernel=3, stride=2, filters=32, stage=stage, block='d')  # TurquiseBlock
    X_shortcut_T_9 = X  # Save for skip connection

    # stage - 10
    stage = 10
    X = OrangeBlock(X_Green_9, kernel=5, filters=128, stage=stage, block='a')  # OrangeBlock
    X_Orange_10 = X
    X = GenericBlock(X, kernel=1, stride=1, filters=192, stage=stage, block='b')  # RedBlock - in order to match sizes
    X_shortcut_R_10 = X  # Save for skip connection

    #################------------------------------------------------------------------- DECONVVVVVVVVVVVVVVVVVV
    # stage - 11
    stage = 11
    X = YellowBlock(X_Orange_10, kernel=5, filters=192, stage=stage, block='a')  # YellowBlock
    X = layers.MaxPool2D(pool_size=2, padding="same")(X)  ## in order to match sizes
    X = layers.Add()([X, X_shortcut_R_10])
    X = YellowBlock(X, kernel=5, filters=64, stage=stage, block='b')  # YellowBlock

    # stage - 12
    stage = 12
    X = GenericBlock(X, kernel=3, stride=1, filters=128, stage=stage, block='a')  # GrayBlock

    # stage - 13
    stage = 13
    X = GenericBlock(X, kernel=1, stride=1, filters=32, stage=stage, block='a')  # RedBlock
    X = layers.MaxPool2D(pool_size=2, padding="same")(X)  ## in order to match sizes
    X = layers.Add()([X, X_shortcut_T_9])
    X = GenericBlock(X, kernel=1, stride=1, filters=128, stage=stage, block='b')  # RedBlock

    #################-------------------------------------------------------------------
    # stage - 14
    stage = 14
    X = YellowBlock(X, kernel=5, filters=64, stage=stage, block='a')  # YellowBlock
    X = layers.Add()([X, X_shortcut_R_7])
    X = YellowBlock(X, kernel=5, filters=64, stage=stage, block='b')  # YellowBlock

    # stage - 15
    stage = 15
    X = GenericBlock(X, kernel=3, stride=1, filters=128, stage=stage, block='a')  # GrayBlock

    # stage - 16
    stage = 16
    X = GenericBlock(X, kernel=1, stride=1, filters=32, stage=stage, block='a')  # RedBlock
    X = layers.MaxPool2D(pool_size=2, padding="same")(X)  ## in order to match sizes
    X = layers.Add()([X, X_shortcut_T_6])
    X = GenericBlock(X, kernel=1, stride=1, filters=128, stage=stage, block='b')  # RedBlock

    #################-------------------------------------------------------------------
    # stage - 17
    stage = 17
    X = YellowBlock(X, kernel=5, filters=32, stage=stage, block='a')  # YellowBlock
    X = layers.Add()([X, X_shortcut_R_4])
    X = YellowBlock(X, kernel=5, filters=32, stage=stage, block='b')  # YellowBlock

    # stage - 18
    stage = 18
    X = GenericBlock(X, kernel=3, stride=1, filters=64, stage=stage, block='a')  # GrayBlock

    # stage - 19
    stage = 19
    X = GenericBlock(X, kernel=1, stride=1, filters=16, stage=stage, block='a')  # RedBlock
    X = layers.MaxPool2D(pool_size=2, padding="same")(X)  ## in order to match sizes
    X = layers.Add()([X, X_shortcut_T_3])
    X = GenericBlock(X, kernel=1, stride=1, filters=64, stage=stage, block='b')  # RedBlock

    #################-------------------------------------------------------------------
    # stage - 20
    stage = 20
    X = YellowBlock(X, kernel=5, filters=16, stage=stage, block='a')  # YellowBlock
    X = layers.Add()([X, X_shortcut_R_1])
    # X = YellowBlock(X,kernel=5,filters=16, stage=stage, block='b') #YellowBlock
    X = YellowBlock(X, kernel=5, filters=8, stage=stage, block='b')  # YellowBlock ##REDUCED FILTERS - RESOURCE LIMIT

    # stage - 21
    stage = 21
    # X = GenericBlock(X,kernel=3,stride=1,filters=32, stage=stage, block='a') #GrayBlock
    X = GenericBlock(X, kernel=3, stride=1, filters=16, stage=stage,
                     block='a')  # GrayBlock ##REDUCED FILTERS - RESOURCE LIMIT

    # stage - 22
    stage = 22
    X = GenericBlock(X, kernel=1, stride=1, filters=32, stage=stage, block='a')  # RedBlock
    #################-------------------------------------------------------------------
    # stage - 23
    stage = 23
    X = GreenBlock(X, kernel=1, filters=1, stage=stage, block='a')  # GreenBlock
    X = layers.MaxPool2D(pool_size=2, padding="same")(X)  ## in order to match sizes

    # Final Addition:
    X = layers.Add()([X, X_input_shortcut])

    return Model(X_input, X, name='BiSkip')


def conv_block(x, filters, activation, kernel_size=(3, 3), strides=(1, 1),
               padding="same", use_bias=True, use_bn=False, use_dropout=False, drop_value=0.5):
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_discriminator_model():
    img_input = layers.Input(shape=(IMAGE_DIM_PARAM, IMAGE_DIM_PARAM, 1))
    # Zero pad the input to make the input images size to (32, 32, 1).
    x = layers.ZeroPadding2D((2, 2))(img_input)
    x = conv_block(
        x,
        64,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(0.2),
        use_dropout=False,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        128,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        256,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        512,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
    )

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)

    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


class WGAN(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            discriminator_extra_steps=3,
            gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        # print("data",data[0])
        data = data[0]
        real_images = data[0]
        blurr_images = data[1]

        # Get the batch size
        batch_size = real_images.shape[0]

        # print("real_shape",real_images,real_images.shape)
        # print("blurr_images_shape",real_images,blurr_images.shape)
        # print("batch_size",batch_size)

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # # Get the latent vector
            # random_latent_vectors = tf.random.normal(
            #     shape=(batch_size, self.latent_dim)
            # )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(blurr_images, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(blurr_images, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


def apply_saved_weights(model, hdf5_file):
    model.built = True
    model.load_weights(hdf5_file, by_name=False)


def Model_Eval(model, Blur_Valid_DS, Sharp_Valid_DS, Num_of_figs):
    generated_images = model.generator(Blur_Valid_DS)  # blur_valid_ds
    generated_images = (generated_images * 127.5) + 127.5
    # print("generated_images.shape",generated_images.shape)

    j = 0
    n_cols = 6
    n_rows = int(Num_of_figs / 2)
    plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    for i in range(0, Num_of_figs, 2):
        img = generated_images[i].numpy()
        img = keras.preprocessing.image.array_to_img(img)
        # img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
        blur_img = Blur_Valid_DS[i]
        blur_img = keras.preprocessing.image.array_to_img(blur_img)
        # blur_img.save("blur_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
        sharp_img = Sharp_Valid_DS[i]
        sharp_img = keras.preprocessing.image.array_to_img(sharp_img)
        # sharp_img.save("Ground_truth_img_{i}_{epoch}.png".format(i=i, epoch=epoch))

        img2 = generated_images[i + 1].numpy()
        img2 = keras.preprocessing.image.array_to_img(img2)
        # img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
        blur_img2 = Blur_Valid_DS[i + 1]
        blur_img2 = keras.preprocessing.image.array_to_img(blur_img2)
        # blur_img.save("blur_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
        sharp_img2 = Sharp_Valid_DS[i + 1]
        sharp_img2 = keras.preprocessing.image.array_to_img(sharp_img2)
        # sharp_img.save("Ground_truth_img_{i}_{epoch}.png".format(i=i, epoch=epoch))

        ax = plt.subplot(n_rows, n_cols, j + 1)
        plt.imshow(blur_img, cmap='gray')
        plt.axis('off')
        if i == 0:
            ax.set_title("Motion Blurred", fontsize=20)

        ax = plt.subplot(n_rows, n_cols, j + 2)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if i == 0:
            ax.set_title("Sharp  - Reconstructed", fontsize=20)

        ax = plt.subplot(n_rows, n_cols, j + 3)
        plt.imshow(sharp_img, cmap='gray')
        plt.axis('off')
        if i == 0:
            ax.set_title("Sharp  - Ground Truth", fontsize=20)

        ax = plt.subplot(n_rows, n_cols, j + 4)
        plt.imshow(blur_img2, cmap='gray')
        plt.axis('off')
        if i == 0:
            ax.set_title("Motion Blurred", fontsize=20)

        ax = plt.subplot(n_rows, n_cols, j + 5)
        plt.imshow(img2, cmap='gray')
        plt.axis('off')
        if i == 0:
            ax.set_title("Sharp  - Reconstructed", fontsize=20)

        ax = plt.subplot(n_rows, n_cols, j + 6)
        plt.imshow(sharp_img2, cmap='gray')
        plt.axis('off')
        if i == 0:
            ax.set_title("Sharp  - Ground Truth", fontsize=20)

        j = j + 6
    plt.show()
    # plt.savefig("Model_Evaluation.png", dpi=300)
    plt.close("all")


def select_ROI(frame):
    """
    Select a ROI and then press SPACE or ENTER button!
    Cancel the selection process by pressing c button!
    Finish the selection process by pressing ESC button!

    """
    fromCenter = False
    ROIs = cv.selectROIs('Select ROIs', frame, fromCenter)

    return ROIs


def record_vid(frameA, frameB, frameC, frameD):
    # print("shapes")
    # print(frameA.shape,frameB.shape,frameC.shape,frameD.shape)
    output = np.zeros((h * 2, w * 2, 3), dtype="uint8")
    output[0:h, 0:w] = cv.cvtColor(np.expand_dims(frameA, -1), cv.COLOR_GRAY2BGR)
    output[0:h, w:w * 2] = cv.cvtColor(frameB.astype(np.uint8), cv.COLOR_GRAY2BGR)
    output[h:h * 2, w:w * 2] = cv.cvtColor(np.expand_dims(frameC, -1), cv.COLOR_GRAY2BGR)
    output[h:h * 2, 0:w] = cv.cvtColor(frameD.astype(np.uint8), cv.COLOR_GRAY2BGR)

    writer.write(output.astype("uint8"))
    cv.imshow("Output", output)
    return


def create_folder():
    image_folder_name = '\images'
    path = os.getcwd()
    Newdir_path = path + image_folder_name
    try:
        os.mkdir(Newdir_path)
    except OSError:
        print("Creation of the directory %s failed" % Newdir_path)
    os.chdir(Newdir_path)
    print("current dir.", os.getcwd())


def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


def extract_patches(img, patch_size, model):
    # wgan receives (BATCH_SIZE,128,128,1) input
    img = tf.expand_dims(img, axis=0)
    img = tf.expand_dims(img, axis=-1)
    # print(img.shape)

    # Patches:
    patches = tf.image.extract_patches(images=img,
                                       sizes=[1, patch_size, patch_size, 1],
                                       strides=[1, patch_size, patch_size, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='SAME')
    # print("patches.shape",patches.shape)

    # canves image of zeros - NOTE that because of the padding the result canvas shape is NOT equal to the orinigal image input and therfore we need center crop at the end of the process
    canvas = np.zeros(shape=[patches.shape[1] * patch_size, patches.shape[2] * patch_size, 1])
    # print("canvas.shape",canvas.shape)

    # Pipeline for each patch (reshape -> normalize -> compute deblure (with wGAN) -> stitch the result in canvas):
    # plt.figure(figsize=(10, 10))
    for imgs in patches:
        count = 0
        for r in range(patches.shape[1]):
            for c in range(patches.shape[2]):
                patched_img = tf.reshape(imgs[r, c], shape=(1, patch_size, patch_size, 1)).numpy().astype("uint8")
                # print("patched_img.shape",patched_img.shape)
                deblured_img = model.generator(patched_img / 255.0)  # Deblure GAN recieves normalized imgs!!!!
                canvas[r * patch_size:r * patch_size + patch_size,
                c * patch_size:c * patch_size + patch_size] = deblured_img
                # ax = plt.subplot(patches.shape[1], patches.shape[2], count + 1)
                # plt.imshow(patched_img[0],cmap="gray")
                count += 1

    # plt.show()
    # plt.imshow(canvas,cmap="gray")
    # plt.show()

    # Final crop: NOTE that because of the padding the result canvas shape is NOT equal to the orinigal image input and therfore we need center crop at the end of the process
    crop = layers.CenterCrop(img.shape[1], img.shape[2])
    canvas = crop(canvas)
    canvas = canvas.numpy()
    # print("canvas.shape", canvas.shape)
    # print("max",np.amax(canvas))
    # print("min", np.amin(canvas))

    # Note that wgan output is not in the range of 0,255 and therfore we need mapping:
    canvas = interval_mapping(canvas, np.amin(canvas), np.amax(canvas), 0, 255.0)
    canvas = canvas.astype(np.uint8)
    # print("canvas.shape", canvas.shape)
    # plt.imshow(canvas,cmap="gray")
    # plt.show()

    return canvas


# Load Wgan Model:
BiSkip = BiSkip(IMAGE_DIM_PARAM)
# BiSkip.summary()
d_model = get_discriminator_model()
# d_model.summary()


# Instantiate the WGAN model.
wgan = WGAN(
    discriminator=d_model,
    generator=BiSkip,
    latent_dim=(IMAGE_DIM_PARAM, IMAGE_DIM_PARAM, 1),
    discriminator_extra_steps=3,
)

# Start training the model.
print(Blur_Train_DS.shape)
print(Sharp_Train_DS.shape)
print(Blur_Valid_DS.shape)
print(Sharp_Valid_DS.shape)

# Use if loading weights from past runs is desired
apply_saved_weights(wgan, hdf_path)
# Model_Eval(wgan, Blur_Valid_DS,Sharp_Valid_DS, 10)
# print("GOOD INPUT--------------------",Blur_Valid_DS[0])


firstFrame = None

count = 0
######------------######------------######------------######------------######------------

# Fill Analysis Parameters Here:
print("Select Video File to analyze.....")
root = tk.Tk()
root.withdraw()
cap_path = filedialog.askopenfilename()
print("cap_path", cap_path)
# cap_path = os.path.join(os.getcwd(),cap_path)
cap = cv.VideoCapture(cap_path)
# cap = cv.VideoCapture(0)
######------------######------------######------------######------------######------------


# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv.VideoWriter_fourcc(*'MJPG')
writer = None
(h, w) = (None, None)
zeros = None

if save_frames is True:
    create_folder()

global ROIs
ret, frame1 = cap.read()

length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # frame count
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv.CAP_PROP_FPS)
# print(length,width,height,fps)

if START_VID_FROM_TIME == 0:
    start_from_frame = 1
else:
    start_from_frame = START_VID_FROM_TIME * fps_input

cap.set(cv.CAP_PROP_POS_FRAMES, start_from_frame)

ROIs = select_ROI(frame1)
print(ROIs, type(ROIs))

# ROI_1 = frame1[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]
# cv.imshow('1', ROI_1)
# cv.waitKey(0)
# cv.destroyAllWindows()


frame1 = frame1[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]

prvs_ff = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)  # frame1 frame in Gray
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
ret = True
while (ret):
    ret, frame2 = cap.read()
    curr_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
    curr_time = curr_frame / fps_input
    END_TIME_FLAG = curr_time > STOP_AT_TIME

    # check if ret is False
    if ret is False or END_TIME_FLAG:
        print("Done!\n")
        cap.release()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()
        break

    frame2 = frame2[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]
    if firstFrame is None:
        firstFrame = prvs_ff
        firstFrame_deblured = extract_patches(firstFrame, IMAGE_DIM_PARAM, wgan)
        cv.imwrite('firstframe.jpg', firstFrame)
        cv.imwrite('firstFrame_deblured.jpg', firstFrame_deblured)
        print("Reading First Frame:")
        print("Frame #", "First Frame")

        # Capture Writer Settings
        if writer is None and save_frames is True:
            (h, w) = firstFrame.shape[:2]
            output = 'GAN_Deblur.avi'
            fps = fps_input
            # fps = int(3 * wait_key / 1000)
            writer = cv.VideoWriter(output, fourcc, 5, (w * 2, h * 2), True)
            zeros = np.zeros((h, w), dtype="uint8")

    if not ret:
        cap.release()
        cv.destroyAllWindows()
        print('Done!\n')
    feed = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Operation On Frame -------

    # get the start time
    st = time.time()
    feed_deblurred = extract_patches(feed, IMAGE_DIM_PARAM, wgan)
    # get the execution time
    print('Execution time of deblurring task:', time.time() - st, 'seconds')

    # Frame Counter
    print('Read a new frame: ', feed.shape)
    print('Frame #', str(count) + '\n')
    count += 1

    # Show The Frames
    if save_frames is not True:
        print("Displaying frames...")
        cv.imshow('first frame', firstFrame)
        cv.imshow('feed', feed)
        cv.imshow('firstFrame_deblured', firstFrame_deblured)
        cv.imshow('feed_deblurred', feed_deblurred)

    # Saving The frames as JPEG files and Output Video
    if save_frames is True:
        print("saving frames...")
        count -= 1
        record_vid(feed, feed_deblurred, firstFrame, firstFrame_deblured)
        count += 1

    key = cv.waitKey(wait_key) & 0xFF

    # Release
    if key == ord("q"):
        cap.release()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()
        break
    elif key == ord('s'):
        cv.imwrite('feedsave.png', feed)
    prvs = next
    continue
