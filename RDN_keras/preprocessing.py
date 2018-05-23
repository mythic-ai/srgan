import numpy as np
import os
import re
from random import shuffle, randint

from imageio import imread

def load_file_list(path, regx):
    files = [f for f in os.listdir(path) if re.search(regx, f)]
    print('Found {} files in {}'.format(len(files), path))
    return files

def DIV2K_RAM_generate(train, batch_size, crop_size):
    """Generate crops of the DIV2K images from the NTIRE 2018 challenge.
       This version preloads the entire dataset into memory, greatly speeding up training."""

    scale = 4
    data_path = '../data/ntire2018/'
    if train:
        lr_dir = data_path + 'DIV2K_train_LR_wild/'
        hr_dir = data_path + 'DIV2K_train_HR/'
    else:
        lr_dir = data_path + 'DIV2K_valid_LR_wild/'
        hr_dir = data_path + 'DIV2K_valid_HR/'

    lr_files = load_file_list(lr_dir, '^[^.].*\.png')
    lr_files.sort()
    lr_imgs = []
    hr_imgs = []
    index_pairs = []
    last_hr_file = ''
    for i, lr_file in enumerate(lr_files):
        if i % 100 == 0:
            print('Loaded {}/{} images'.format(i, len(lr_files)))

        # Load low-res image into memory
        lr_img = imread(lr_dir + lr_file)
        lr_imgs.append(lr_img)

        # Load high-res image into memory, but only if it's a new image
        hr_file = lr_file[:4] + '.png'
        if last_hr_file != hr_file:
            last_hr_file = hr_file
            hr_img = imread(hr_dir + hr_file)
            hr_imgs.append(hr_img)

        # Track which pair of LR/HR images correspond to one another
        index_pairs.append((i, len(hr_imgs) - 1))

        # Check image dimensions
        if crop_size is not None:
            assert lr_img.shape[0] >= crop_size
            assert lr_img.shape[1] >= crop_size
        assert lr_img.shape[0] * scale == hr_img.shape[0]
        assert lr_img.shape[1] * scale == hr_img.shape[1]

    print("Loaded {} low-res and {} high-res images into memory!".format(len(lr_imgs), len(hr_imgs)))

    lr_batch = []
    hr_batch = []

    while True:
        if train:
            # Shuffle during training to decorrelate gradient signals.
            shuffle(index_pairs)
        else:
            # Ensure the validation set is deterministic.
            lr_batch = []
            hr_batch = []
            assert crop_size is None

        for i, j in index_pairs:
            # Read the image
            lr_img = lr_imgs[i]
            hr_img = hr_imgs[j]

            # Crop a random patch out of the image
            if crop_size is not None:
                crop_t = randint(0, lr_img.shape[0] - crop_size)
                crop_b = crop_t + crop_size
                crop_l = randint(0, lr_img.shape[1] - crop_size)
                crop_r = crop_l + crop_size
                lr_img = lr_img[crop_t:crop_b, crop_l:crop_r]
                hr_img = hr_img[crop_t * scale:crop_b * scale, crop_l * scale:crop_r * scale]

            # Normalize the image between [-1, 1]
            lr_img = lr_img / 127.5 - 1
            hr_img = hr_img / 127.5 - 1

            # Apply nondetermimistic data augmentation, in the form of rotations and flips
            if not train:
                orientation = randint(0, 7)
                if orientation >= 4:
                    lr_img = np.flip(lr_img, 0)
                    hr_img = np.flip(hr_img, 0)
                if orientation % 4 != 0:
                    lr_img = np.rot90(lr_img, orientation % 4, [0, 1])
                    hr_img = np.rot90(hr_img, orientation % 4, [0, 1])

            # Add the LR and HR patches to the current batch.
            lr_batch.append(lr_img)
            hr_batch.append(hr_img)
            if len(hr_batch) == batch_size:
                yield np.array(lr_batch), np.array(hr_batch)
                lr_batch = []
                hr_batch = []

"""Ignore everything after this line :)"""

def DIV2K_disk_generate(train, batch_size, crop_size):
    """Generate crops of the DIV2K images from the NTIRE 2018 challenge.
       This version loads individual batches from disk, which saves memory and start-up time but is very slow."""

    scale = 4
    data_path = '../data/ntire2018/'
    if train:
        lr_dir = data_path + 'DIV2K_train_LR_wild/'
        hr_dir = data_path + 'DIV2K_train_HR/'
    else:
        lr_dir = data_path + 'DIV2K_valid_LR_wild/'
        hr_dir = data_path + 'DIV2K_valid_HR/'

    lr_files = load_file_list(lr_dir, '^[^.].*\.png')

    lr_batch = []
    hr_batch = []

    while True:
        if train:
            # Shuffle during training to decorrelate gradient signals.
            shuffle(lr_files)
        else:
            # Ensure the validation set is deterministic.
            lr_batch = []
            hr_batch = []

        for lr_file in lr_files:
            hr_file = lr_file[:4] + '.png'

            # Read the image
            lr_img = imread(lr_dir + lr_file)
            hr_img = imread(hr_dir + hr_file)

            # Crop a random patch out of the image
            if crop_size is not None:
                crop_t = randint(0, lr_img.shape[0] - crop_size)
                crop_b = crop_t + crop_size
                crop_l = randint(0, lr_img.shape[1] - crop_size)
                crop_r = crop_l + crop_size
                lr_img = lr_img[crop_t:crop_b, crop_l:crop_r]
                hr_img = hr_img[crop_t*scale:crop_b*scale, crop_l*scale:crop_r*scale]

            lr_img = lr_img / 127.5 - 1
            hr_img = hr_img / 127.5 - 1
            if not train:
                orientation = randint(0, 7)
                if orientation >= 4:
                    lr_img = np.flip(lr_img, 0)
                    hr_img = np.flip(hr_img, 0)
                if orientation % 4 != 0:
                    lr_img = np.rot90(lr_img, orientation % 4, [0, 1])
                    hr_img = np.rot90(hr_img, orientation % 4, [0, 1])

            # Add the LR and HR patches to the current batch.
            lr_batch.append(lr_img)
            hr_batch.append(hr_img)
            if len(hr_batch) == batch_size:
                yield np.array(lr_batch), np.array(hr_batch)
                lr_batch = []
                hr_batch = []