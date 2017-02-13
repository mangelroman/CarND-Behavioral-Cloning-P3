import os
import csv
import numpy as np

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
            samples.append([line[0].strip(), steering_center, False])
            samples.append([line[1].strip(), steering_left, False])
            samples.append([line[2].strip(), steering_right, False])
            samples.append([line[0].strip(), -steering_center, True])
            samples.append([line[1].strip(), -steering_left, True])
            samples.append([line[2].strip(), -steering_right, True])

    return samples
