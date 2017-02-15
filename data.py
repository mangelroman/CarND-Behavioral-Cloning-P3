import os
import csv
import cv2
import numpy as np

def transform_image(image, transtype):
    if (transtype == 'flip'):
        return cv2.flip(image, 1)
    return image

def load_samples(datadir, correction=0.22):
    samples = []
    logpath = os.path.join(datadir, 'driving_log.csv')
    with open(logpath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            steering_center = float(line[3])

            # Randomly discard 0 steering samples
            if steering_center == 0:
                discard = np.random.choice([True, False], p=[0.8, 0.2])
                if discard:
                    continue

            # create adjusted steering measurements for the side camera images
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # add images and angles to data set
            samples.append([line[0].strip(), steering_center, ''])
            samples.append([line[1].strip(), steering_left, ''])
            samples.append([line[2].strip(), steering_right, ''])
            samples.append([line[0].strip(), -steering_center, 'flip'])
            samples.append([line[1].strip(), -steering_left, 'flip'])
            samples.append([line[2].strip(), -steering_right, 'flip'])

    return samples
