import os
import csv
import pickle
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

image_shape = (160, 320, 3)
pimage_shape = (image_shape[0] // 2, image_shape[1] // 2, image_shape[2])

crop_height = (60 // 2, 36 // 2)
crop_width = (0, 0)

cimage_shape = (pimage_shape[0] - sum(crop_height), pimage_shape[1] - sum(crop_width), pimage_shape[2])

def preprocess_image(image, flip=False):
    image = cv2.resize(np.asarray(image), (pimage_shape[1], pimage_shape[0]), interpolation=cv2.INTER_AREA)
    if (flip):
        image = cv2.flip(image, 1)
    return image / 127.5 - 1.0

def generator(samples, batch_size=256):
    n_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, n_samples, batch_size):
            samples_batch = samples[offset:offset+batch_size]
            n_batch = len(samples_batch)
            y_batch = np.empty(n_batch)
            X_batch = np.empty([n_batch, pimage_shape[0], pimage_shape[1], pimage_shape[2]])
            for i, sample in enumerate(samples_batch):
                image = cv2.imread(data_folder + sample[0])
                X_batch[i] = preprocess_image(image, sample[2])
                y_batch[i] = sample[1]
            yield X_batch, y_batch

if __name__ == '__main__':
    samples = []

    data_folder = "./data/"

    with open(data_folder + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            steering_center = float(line[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # add images and angles to data set
            samples.append([line[0].strip(), steering_center, False])
            samples.append([line[1].strip(), steering_left, False])
            samples.append([line[2].strip(), steering_right, False])
            samples.append([line[0].strip(), -steering_center, True])
            samples.append([line[1].strip(), -steering_left, True])
            samples.append([line[2].strip(), -steering_right, True])

    #X_samples_aug
    #for image, steering in zip(X_samples, y_samples):
    print("Shuffling {} samples...".format(len(samples)))
    shuffle(samples)

    print("Splitting training and validation sets...".format(len(samples)))
    samples_train, samples_val = train_test_split(samples, test_size=0.2, random_state=2017)

    # compile and train the model using the generator function
    train_generator = generator(samples_train, batch_size=32)
    validation_generator = generator(samples_val, batch_size=32)

    from keras.models import Sequential
    from keras.layers import Lambda, Cropping2D, Dense, Input
    from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Cropping2D(cropping=(crop_height, crop_width), input_shape=pimage_shape))
    model.add(Convolution2D(8,5,5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(16,5,5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32,5,5, activation='relu', border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
                        samples_per_epoch=len(samples_train),
                        validation_data=validation_generator,
                        nb_val_samples=len(samples_val),
                        nb_epoch=5)

    model.save('model.h5')
