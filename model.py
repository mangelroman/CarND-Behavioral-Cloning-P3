import os
import argparse
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Dense, ELU
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout
from keras import backend as K

from data import load_samples

image_shape = (160, 320, 3)
pimage_shape = (image_shape[0] // 2, image_shape[1] // 2, image_shape[2])

crop_height = (60 // 2, 36 // 2)
crop_width = (10, 10)

cimage_shape = (pimage_shape[0] - sum(crop_height), pimage_shape[1] - sum(crop_width), pimage_shape[2])

def preprocess_image(image, flip=False):
    image = cv2.resize(image, (pimage_shape[1], pimage_shape[0]), interpolation=cv2.INTER_AREA)
    if (flip):
        image = cv2.flip(image, 1)
    return image

def normalize_image(image):
    cmin = K.min(image, axis=[1,2,3], keepdims=True)
    cmax = K.max(image, axis=[1,2,3], keepdims=True)

    return 2 * (image - cmin) / (cmax - cmin) - 1.0

def generator(samples, datadir, batch_size):
    samples = shuffle(samples)
    n_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, n_samples, batch_size):
            samples_batch = samples[offset:offset+batch_size]
            n_batch = len(samples_batch)
            y_batch = np.empty(n_batch, dtype=float)
            X_batch = np.empty([n_batch, pimage_shape[0], pimage_shape[1], pimage_shape[2]], dtype='uint8')
            for i, sample in enumerate(samples_batch):
                imagepath = os.path.join(datadir, sample[0])
                image = cv2.imread(imagepath)
                X_batch[i] = preprocess_image(image, sample[2])
                y_batch[i] = sample[1]
            yield X_batch, y_batch

def get_manet_model():
    model = Sequential()
    model.add(Cropping2D(cropping=(crop_height, crop_width), input_shape=pimage_shape))
    model.add(Lambda(normalize_image))
    model.add(Convolution2D(8,5,5, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(16,5,5, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32,5,5, activation='relu', border_mode='same'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.summary()
    return model

def get_nvidia_model():
    model = Sequential()
    model.add(Cropping2D(cropping=(crop_height, crop_width), input_shape=pimage_shape))
    model.add(Lambda(normalize_image))
    # NVIDIA MODEL ARCHITECTURE STARTS HERE
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    return model

def get_comma_model():
    model = Sequential()
    model.add(Cropping2D(cropping=(crop_height, crop_width), input_shape=pimage_shape))
    model.add(Lambda(normalize_image))
    # COMMA.AI MODEL ARCHITECTURE STARTS HERE
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
    model.summary()
    return model

def get_model(name):
    return {'manet' : get_manet_model, 'nvidia' : get_nvidia_model, 'comma' : get_comma_model} [name] ()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Behavioral Cloning Project")
    parser.add_argument('--datadir', type=str, default='data', help="Directory containing training data samples")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument('--batch', type=int, default=256, help="Training batch size")
    parser.add_argument('--model', type=str, default='manet', help="Training batch size")
    args = parser.parse_args()

    samples = load_samples(args.datadir)

    print("Shuffling {} samples...".format(len(samples)))
    shuffle(samples)

    print("Splitting training and validation sets...")
    samples_train, samples_val = train_test_split(samples, test_size=0.2, random_state=2017)

    # compile and train the model using the generator function
    train_generator = generator(samples_train, args.datadir, args.batch)
    validation_generator = generator(samples_val, args.datadir, args.batch)

    model = get_model(args.model)

    print("Training model with {} parameters...".format(model.count_params()))
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
                        samples_per_epoch=len(samples_train),
                        validation_data=validation_generator,
                        nb_val_samples=len(samples_val),
                        nb_epoch=args.epochs)

    model.save('model.h5')
