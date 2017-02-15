import os
import argparse
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Dense, Dropout, ELU
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from data import load_samples, transform_image

imgsize = (160, 320, 3)

croph = (60, 36)
cropw = (0, 0)

def preprocess_image(image, newshape):
    h, w, c = image.shape
    image = image[croph[0]:h-croph[1], cropw[0]:w-cropw[1],:]
    return cv2.resize(image, (newshape[1], newshape[0]), interpolation=cv2.INTER_AREA)

def normalize(images):
    # Normalize to keep values in the [-1, 1] range
    return images.astype(np.float32) / 127.5 - 1.0

def generator(samples, datadir, batch_size, input_shape):
    samples = shuffle(samples)
    n_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, n_samples, batch_size):
            samples_batch = samples[offset:offset+batch_size]
            n_batch = len(samples_batch)
            y_batch = np.empty(n_batch, dtype=float)
            X_batch = np.empty([n_batch, input_shape[0], input_shape[1], input_shape[2]], dtype='uint8')
            for i, sample in enumerate(samples_batch):
                imagepath = os.path.join(datadir, sample[0])
                image = cv2.imread(imagepath)
                image = transform_image(image, sample[2])
                X_batch[i] = preprocess_image(image, input_shape)
                y_batch[i] = sample[1]
            X_batch = normalize(X_batch)
            yield X_batch, y_batch

def get_manet_model():
    input_shape = (64, 64, 3)
    model = Sequential()
    #model.add(Cropping2D(cropping=(crop_height, crop_width), input_shape=input_shape))
    #model.add(Lambda(normalize, input_shape=input_shape))
    model.add(Convolution2D(8,5,5, input_shape=input_shape, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16,5,5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32,5,5, activation='relu', border_mode='valid'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.summary()
    return model, input_shape

def get_nvidia_model():
    input_shape = (66, 200, 3)
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, input_shape=input_shape, border_mode='valid', subsample=(2,2), activation='relu'))
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
    model.summary()
    return model, input_shape

def get_comma_model():
    input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Convolution2D(16, 8, 8, input_shape=input_shape, subsample=(4, 4), border_mode="same"))
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
    return model, input_shape

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
    #samples = load_samples("data_udacity")
    #samples.extend(load_samples("data_udacity"))

    print("Shuffling {} samples...".format(len(samples)))
    shuffle(samples)

    print("Splitting training and validation sets...")
    samples_train, samples_val = train_test_split(samples, test_size=0.2, random_state=17)

    # compile and train the model using the generator function
    model, input_shape = get_model(args.model)
    model.compile(loss='mse', optimizer='adam')

    #with open(args.model + ".json", "w") as file:
    #    file.write(model.to_json())

    print("Training model with {} parameters...".format(model.count_params()))
    train_generator = generator(samples_train, args.datadir, args.batch, input_shape)
    validation_generator = generator(samples_val, args.datadir, args.batch, input_shape)

    # Early stopping to help test longer epochs
    earlystopper = EarlyStopping(monitor='val_loss', patience=4)

    # saves the model weights after each epoch if the validation loss decreased
    checkpointer = ModelCheckpoint(filepath=args.model + ".h5", verbose=0, save_best_only=True)# , save_weights_only=True)

    model.fit_generator(train_generator,
                        samples_per_epoch=len(samples_train),
                        validation_data=validation_generator,
                        nb_val_samples=len(samples_val),
                        nb_epoch=args.epochs,
                        callbacks=[checkpointer, earlystopper])

    K.clear_session()
