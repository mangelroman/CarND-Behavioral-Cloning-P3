import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Dense, Dropout, ELU
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils.visualize_util import plot
from keras import backend as K

from data import Sample, load_samples, translate_samples, flip_samples, shadow_samples

IMAGE_SHAPE = (160, 320, 3)
CROP_HEIGHT = (58, 22)
CROP_WIDTH = (0, 0)

REG_PARAM = 0
LEARNING_RATE = 0.001

def preprocess_image(image, newshape):
    h, w, c = image.shape
    image = image[CROP_HEIGHT[0]:h-CROP_HEIGHT[1], CROP_WIDTH[0]:w-CROP_WIDTH[1],:]
    return cv2.resize(image, (newshape[1], newshape[0]), interpolation=cv2.INTER_AREA)

# def minmax_normalize(images):
#     cmin = K.min(images, axis=[1, 2, 3], keepdims=True)
#     cmax = K.max(images, axis=[1, 2, 3], keepdims=True)
#     return 2 * (images - cmin) / (cmax - cmin) - 1.0

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
                X_batch[i] = preprocess_image(sample.get_image(), input_shape)
                y_batch[i] = sample.get_steering()
            yield X_batch, y_batch

def get_simple_model():
    input_shape = (40, 40, 3)
    model = Sequential()
    model.add(Lambda(normalize, input_shape=input_shape))
    model.add(Convolution2D(8,5,5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16,5,5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32,3,3, activation='relu', border_mode='valid'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu')) #, W_regularizer=l2(REG_PARAM)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu')) #, W_regularizer=l2(REG_PARAM)))
    model.add(Dense(1))
    return model, input_shape

def get_nvidia_model():
    input_shape = (66, 200, 3)
    model = Sequential()
    model.add(Lambda(normalize, input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1164, W_regularizer=l2(REG_PARAM), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100, W_regularizer=l2(REG_PARAM), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, W_regularizer=l2(REG_PARAM), activation='relu'))
    model.add(Dense(10, W_regularizer=l2(REG_PARAM), activation='relu'))
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
    return {'simple' : get_simple_model, 'nvidia' : get_nvidia_model, 'comma' : get_comma_model} [name] ()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Behavioral Cloning Project")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train the model")
    parser.add_argument('--batch', type=int, default=256, help="Training batch size")
    parser.add_argument('--model', type=str, default='simple', help="Training batch size")
    parser.add_argument('--patience', type=int, default=4, help="Number of epochs to wait for improvement before early stop")
    parser.add_argument('--valdir', type=str, default='', help="Directory containing validation data samples")
    parser.add_argument('traindir', type=str, default='data_train', help="Directory containing training data samples")
    parser.add_argument('outfile', type=str, default='model.h5', help="Training batch size")
    args = parser.parse_args()

    # compile and train the model using the generator function
    model, input_shape = get_model(args.model)
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    model.summary()
    #plot(model, show_shapes=True, to_file="{}.png".format(args.model))

    print("Loading training set... ", end='')
    samples_train = load_samples(args.traindir)
    print("{} samples".format(len(samples_train)))

    if (len(args.valdir) > 0):
        print("Loading validation set... ", end='')
        samples_val = load_samples(args.valdir)
        print("{} samples".format(len(samples_val)))
    else:
        print("Splitting training and validation sets... ", end='')
        samples_train = shuffle(samples_train)
        samples_train, samples_val = train_test_split(samples_train, test_size=0.2, random_state=43)
        print("{}/{} samples".format(len(samples_train), len(samples_val)))

    print("Augmenting training set... {} -> ".format(len(samples_train)), end='')
    translate_samples(samples_train)
    flip_samples(samples_train)
    shadow_samples(samples_train)
    print("{} samples".format(len(samples_train)))

    print("Augmenting validation set... {} -> ".format(len(samples_val)), end='')
    flip_samples(samples_val)
    print("{} samples".format(len(samples_val)))

    print("Training model with {} parameters...".format(model.count_params()))
    train_generator = generator(samples_train, args.batch, input_shape)
    validation_generator = generator(samples_val, args.batch, input_shape)

    # Early stopping to help test longer epochs
    earlystopper = EarlyStopping(monitor='val_loss', patience=args.patience)

    # Saves the model after each epoch if the validation loss decreased
    checkpointer = ModelCheckpoint(filepath=args.outfile, verbose=0, save_best_only=True)

    # Saves TensorBoard data for model visualization
    tensorboarder = TensorBoard(write_images=True)

    history_object = model.fit_generator(
        train_generator,
        samples_per_epoch=len(samples_train),
        validation_data=validation_generator,
        nb_val_samples=len(samples_val),
        nb_epoch=args.epochs,
        callbacks=[checkpointer, earlystopper, tensorboarder])

    # Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    K.clear_session()
