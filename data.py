import os
import csv
import cv2
import numpy as np

def random_shadow(image):
    h, w, _ = image.shape
    shadow_image = np.copy(image)
    [x1, x2] = np.random.choice(w, 2, replace=False)
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

def get_image(sample):
    image = cv2.imread(sample[0])
    if (sample[2]):
        return cv2.flip(image, 1)
    if (sample[3] != 0):
        image = translate_image(image, sample[3])
    if (sample[4]):
        image = random_shadow(image)
    return image

def get_steering(sample):
    return sample[1]

def get_relative_path(datadir, fullpath):
    dirname,filename = os.path.split(fullpath)
    dirname = os.path.basename(dirname)
    lastpath = os.path.join(datadir, os.path.join(dirname, filename))
    return os.path.relpath(lastpath)

DISCARD_PROB = 0.8
SHADOW_PROB = 0.5
LR_CORRECTION = 0.26
SHIFT_CORRECTION = 0.01
MAX_SHIFT = int(LR_CORRECTION / (2 * SHIFT_CORRECTION))

def load_samples(datadir, augment=True):
    samples = []
    logpath = os.path.join(datadir, 'driving_log.csv')
    with open(logpath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            steering = float(line[3])

            # Randomly discard 0 steering samples
            if steering == 0:
                discard = np.random.choice([True, False], p=[DISCARD_PROB, 1 - DISCARD_PROB])
                if discard:
                    if augment:
                        i = np.random.choice([0, 1, 2])
                        if (i == 1):
                            steering += LR_CORRECTION
                        elif (i == 2):
                            steering -= LR_CORRECTION
                        f = np.random.choice([True, False])
                        if (f):
                            steering = -steering
                        d = np.random.randint(-MAX_SHIFT,MAX_SHIFT)
                        steering = steering + d * SHIFT_CORRECTION
                        s = np.random.choice([True, False], p=[SHADOW_PROB, 1 - SHADOW_PROB])
                        samples.append([get_relative_path(datadir, line[i].strip()), steering, f, d, s])
                    continue

            centerpath = get_relative_path(datadir, line[0].strip())
            leftpath = get_relative_path(datadir, line[1].strip())
            rightpath = get_relative_path(datadir, line[2].strip())

            if augment:
                shadows = np.random.choice([True, False], size=6, p=[SHADOW_PROB, 1 - SHADOW_PROB])
                samples.append([centerpath, steering, False, 0, shadows[0]])
                samples.append([centerpath, -steering, True, 0, shadows[1]])
                samples.append([leftpath,   steering + LR_CORRECTION, False, 0, shadows[2]])
                samples.append([leftpath,   -steering - LR_CORRECTION, True, 0, shadows[3]])
                samples.append([rightpath,  steering - LR_CORRECTION, False, 0, shadows[4]])
                samples.append([rightpath,  -steering + LR_CORRECTION, True, 0, shadows[5]])
            else:
                samples.append([centerpath, steering, False, 0, False])
                samples.append([centerpath, -steering, True, 0, False])



    return samples
