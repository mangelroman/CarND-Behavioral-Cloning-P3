import os
import argparse
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Dense, Dropout, ELU
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K

from data import load_samples, get_image, get_steering

imgsize = (160, 320, 3)

croph = (58, 22)
cropw = (0, 0)

def preprocess_image(image, newshape):
    h, w, c = image.shape
    image = image[croph[0]:h-croph[1], cropw[0]:w-cropw[1],:]
    return cv2.resize(image, (newshape[1], newshape[0]), interpolation=cv2.INTER_AREA)

def minmax_normalize(images):
    cmin = K.min(images, axis=[1, 2, 3], keepdims=True)
    cmax = K.max(images, axis=[1, 2, 3], keepdims=True)
    return 2 * (images - cmin) / (cmax-cmin) - 1.0

def normalize(images):
    # Normalize to keep values in the [-1, 1] range
    return images / 127.5 - 1.0

def generator(samples, batch_size, input_shape):
    samples = shuffle(samples)
    n_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, n_samples, batch_size):
            samples_batch = samples[offset:offset+batch_size]
            n_batch = len(samples_batch)
            y_batch = np.empty(n_batch, dtype=float)
            X_batch = np.empty([n_batch, input_shape[0], input_shape[1], input_shape[2]], dtype='uint8')
            for i, sample in enumerate(samples_batch):
                X_batch[i] = preprocess_image(get_image(sample), input_shape)
                y_batch[i] = get_steering(sample)
            yield X_batch, y_batch

def get_manet_model():
    input_shape = (40, 160, 3)
    model = Sequential()
    #model.add(Cropping2D(cropping=(crop_height, crop_width), input_shape=input_shape))
    model.add(Lambda(normalize, input_shape=input_shape))
    model.add(Convolution2D(8,5,5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16,5,5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32,5,5, activation='relu', border_mode='valid'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(1))
    return model, input_shape

def get_nvidia_model():
    input_shape = (66, 200, 3)
    model = Sequential()
    model.add(Lambda(normalize, input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model, input_shape

def get_comma_model():
    input_shape = (80, 160, 3)
    model = Sequential()
    model.add(Lambda(normalize, input_shape=input_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model, input_shape

def get_model(name):
    return {'manet' : get_manet_model, 'nvidia' : get_nvidia_model, 'comma' : get_comma_model} [name] ()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Behavioral Cloning Project")
    parser.add_argument('--traindir', type=str, default='data_train', help="Directory containing training data samples")
    parser.add_argument('--valdir', type=str, default='data_val', help="Directory containing validation data samples")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument('--batch', type=int, default=256, help="Training batch size")
    parser.add_argument('--model', type=str, default='manet', help="Training batch size")
    args = parser.parse_args()

    samples_train = load_samples(args.traindir, augment=True)
    samples_val = load_samples(args.valdir, augment=False)

    # compile and train the model using the generator function
    model, input_shape = get_model(args.model)
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    print("Training model with {} parameters...".format(model.count_params()))
    train_generator = generator(samples_train, args.batch, input_shape)
    validation_generator = generator(samples_val, args.batch, input_shape)

    # Early stopping to help test longer epochs
    earlystopper = EarlyStopping(monitor='val_loss', patience=4)

    # Saves the model after each epoch if the validation loss decreased
    checkpointer = ModelCheckpoint(filepath=args.model + ".h5", verbose=0, save_best_only=True)

    # Saves
    #tensorboarder = TensorBoard(write_images=True)

    model.fit_generator(train_generator,
                        samples_per_epoch=len(samples_train),
                        validation_data=validation_generator,
                        nb_val_samples=len(samples_val),
                        nb_epoch=args.epochs,
                        callbacks=[checkpointer, earlystopper]) #, tensorboarder])

    K.clear_session()
