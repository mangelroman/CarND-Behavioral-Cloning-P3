import os
import csv
import cv2
import numpy as np
from PIL import Image

DISCARD_PROB = 0.85
TRANSLATE_PROB = 0.5
SHADOW_PROB = 0

LR_CORRECTION = 0.18
MAX_TRANSLATION = 30
TRANS_CORRECTION = LR_CORRECTION / 30

def shadow_image(image, x1, x2):
    h, w, _ = image.shape
    shadow_image = np.copy(image)
    m = (x2 - x1) / h
    b = x1
    for y in range(h):
        c = int(m * y + b)
        shadow_image[y, :c, :] //= 2
    return shadow_image

def translate_image(img, delta):
    y, x, c = img.shape

    M = np.float32([[1,0,delta],[0,1,0]])

    flags=cv2.INTER_LANCZOS4+cv2.WARP_FILL_OUTLIERS
    border=cv2.BORDER_REPLICATE

    return cv2.warpAffine(img,M,(x,y),flags=flags,borderMode=border)

class Sample:
    def __init__(self, path, steering):
        self.path = path
        self.steering = steering
        self.offset = 0
        self.flipped = False
        self.x1_shadow = 0
        self.x2_shadow = 0

    def get_image(self):
        # Reading image with the same method as drive.py to avoid color space inconsistencies
        image = np.asarray(Image.open(self.path))
        if (self.offset != 0):
            image = translate_image(image, self.offset)
        if (self.flipped):
            image = cv2.flip(image, 1)
        if (self.x1_shadow != 0) or (self.x2_shadow != 0):
            image = shadow_image(image, self.x1_shadow, self.x2_shadow)
        return image

    def get_steering(self):
        return self.steering

    def translate(self):
        self.offset = np.random.randint(-(MAX_TRANSLATION+1),(MAX_TRANSLATION+1))
        self.steering += self.offset * TRANS_CORRECTION

    def flip(self):
        self.flipped = not self.flipped
        self.steering = -self.steering

    def shadow(self):
        self.x1_shadow, self.x2_shadow = np.random.choice(320, 2, replace=False)

    def clone(self):
        newsample = Sample(self.path, self.steering)
        newsample.offset = self.offset
        newsample.flipped = self.flipped
        newsample.x1_shadow = self.x1_shadow
        newsample.x2_shadow = self.x2_shadow
        return newsample

def get_relative_path(datadir, fullpath):
    dirname,filename = os.path.split(fullpath)
    dirname = os.path.basename(dirname)
    lastpath = os.path.join(datadir, os.path.join(dirname, filename))
    return os.path.relpath(lastpath)

def translate_samples(samples, prob=TRANSLATE_PROB):
    # Randomly add new translated samples to the samples array
    n_samples = len(samples)
    indexes = np.random.choice(n_samples, size=int(n_samples * prob), replace=False)

    for i in indexes:
        sample = samples[i].clone()
        sample.translate()
        samples.append(sample)
    return samples

def flip_samples(samples):
    # Add flipped samples to the samples array
    n_samples = len(samples)
    for i in range(n_samples):
        sample = samples[i].clone()
        sample.flip()
        samples.append(sample)
    return samples

def shadow_samples(samples, prob=SHADOW_PROB):
    # Randomly add new translated samples to the samples array
    n_samples = len(samples)
    indexes = np.random.choice(n_samples, size=int(n_samples * prob), replace=False)

    for i in indexes:
        samples[i].shadow()
    return samples

def load_samples(datadir, discard=DISCARD_PROB):
    # Read samples from disk and prepare metadata
    samples = []
    samples_zero = []
    logpath = os.path.join(datadir, 'driving_log.csv')
    with open(logpath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            steering = float(line[3])

            centerpath = get_relative_path(datadir, line[0].strip())
            leftpath = get_relative_path(datadir, line[1].strip())
            rightpath = get_relative_path(datadir, line[2].strip())

            # Store samples with 0 steering separately
            if (steering == 0):
                target_array = samples_zero
            else:
                target_array = samples

            target_array.append(Sample(centerpath, steering))
            target_array.append(Sample(leftpath, steering + LR_CORRECTION))
            target_array.append(Sample(rightpath, steering - LR_CORRECTION))

    # Discard 0 steering samples based on selected probability
    n_zero = len(samples_zero)
    indexes = np.random.choice(n_zero, size=int(n_zero * (1 - discard)), replace=False)

    for i in indexes:
        samples.append(samples_zero[i])

    return samples
