import os
import csv
import cv2
import numpy as np
from PIL import Image

DISCARD_PROB = 0.9
SHADOW_PROB = 0
LR_CORRECTION = 0.16
MAX_SHIFT = 20
SHIFT_CORRECTION = LR_CORRECTION / 30

def shadow_image(image, x1, x2):
    h, w, _ = image.shape
    shadow_image = np.copy(image)
    m = h / (x2 - x1)
    b = - m * x1
    for i in range(h):
        c = int((i - b) / m)
        shadow_image[i, :c, :] //= 2
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
        self.flipped = False
        self.offset = 0
        self.x1_shadow = 0
        self.x2_shadow = 0

    def get_image(self):
        # Reading image with the same method as drive.py to avoid color space inconsistencies
        image = np.asarray(Image.open(self.path))
        #image = cv2.imread(self.path)
        if (self.offset != 0):
            image = translate_image(image, self.offset)
        if (self.flipped):
            image = cv2.flip(image, 1)
        if (self.x1_shadow != 0) and (self.x2_shadow != 0):
            image = shadow_image(image, self.x1_shadow, self.x2_shadow)
        return image

    def get_steering(self):
        return self.steering

    def flip(self):
        self.flipped = not self.flipped
        self.steering = -self.steering

    def translate(self):
        self.offset = np.random.randint(-(MAX_SHIFT+1),(MAX_SHIFT+1))
        self.steering += self.offset * SHIFT_CORRECTION

    def shadow(self):
        if(np.random.choice([True, False], p=[SHADOW_PROB, 1 - SHADOW_PROB])):
            self.x1_shadow, self.x2_shadow = np.random.choice(320, 2, replace=False)
        else:
            self.x1_shadow = 0
            self.x2_shadow = 0

    def clone(self):
        newsample = Sample(self.path, self.steering)
        newsample.flipped = self.flipped
        newsample.offset = self.offset
        return newsample

def get_relative_path(datadir, fullpath):
    dirname,filename = os.path.split(fullpath)
    dirname = os.path.basename(dirname)
    lastpath = os.path.join(datadir, os.path.join(dirname, filename))
    return os.path.relpath(lastpath)

def augment_samples(samples):
    # Flip all samples to ensure a perfect balance between right and left steering angles
    n_samples = len(samples)
    for i in range(n_samples):
        sample = samples[i].clone()
        sample.translate()
        sample.shadow()
        samples.append(sample)
    return samples

def balance_samples(samples):
    # Flip all samples to ensure a perfect balance between right and left steering angles
    n_samples = len(samples)
    for i in range(n_samples):
        sample = samples[i].clone()
        sample.flip()
        samples.append(sample)
    return samples

def load_samples(datadir):
    samples = []
    logpath = os.path.join(datadir, 'driving_log.csv')
    with open(logpath) as csvfile:
        reader = csv.reader(csvfile)
        # Skip headings line
        next(reader)
        for line in reader:
            steering = float(line[3])

            # Randomly discard 0 steering samples
            if (steering == 0):
                discard = np.random.choice([True, False], p=[DISCARD_PROB, 1 - DISCARD_PROB])
                if discard:
                    continue

            centerpath = get_relative_path(datadir, line[0].strip())
            leftpath = get_relative_path(datadir, line[1].strip())
            rightpath = get_relative_path(datadir, line[2].strip())

            samples.append(Sample(centerpath, steering))
            samples.append(Sample(leftpath, steering + LR_CORRECTION))
            samples.append(Sample(rightpath, steering - LR_CORRECTION))

    return samples
