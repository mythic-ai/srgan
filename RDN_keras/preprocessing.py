import numpy as np
from random import shuffle, randint

import tensorlayer as tl
from imageio import imread

def DIV2K_generate(train, batch_size, crop_size=32):
    """Generate crops of the DIV2K images from the NTIRE 2018 challenge"""

    scale = 4
    data_path = '../data/ntire2018/'
    if train:
        lr_dir = data_path + 'DIV2K_train_LR_wild/'
        hr_dir = data_path + 'DIV2K_train_HR/'
    else:
        lr_dir = data_path + 'DIV2K_valid_LR_wild/'
        hr_dir = data_path + 'DIV2K_valid_HR/'
    lr_files = tl.files.load_file_list(path=lr_dir, regx='^[^.].*\.png')
    lr_batch = []
    hr_batch = []

    while True:
        if train:
            # Decorrelate gradient signals during training.
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

            # TODO: add normalization, flipping, and 90-deg rotations as in the paper.
            # TODO: check if data reading is a bottleneck.
            # TODO: add additional data augmentation and compare results on realistic data.

            # Add the LR and HR patches to the current batch.
            lr_batch.append(lr_img)
            hr_batch.append(hr_img)
            if len(hr_batch) == batch_size:
                yield np.array(lr_batch), np.array(hr_batch)
                lr_batch = []
                hr_batch = []