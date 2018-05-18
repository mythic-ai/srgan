#! /usr/bin/python
# -*- coding: utf8 -*-

import os
from imageio import imread
from timeit import default_timer as timer

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from model import RDN147
from preprocessing import DIV2K_RAM_generate, DIV2K_disk_generate

###====================== HYPER-PARAMETERS ===========================###
BATCH_SIZE = 16
N_EPOCHS = 200

def train(weights_file=None):
    model = RDN147()
    if weights_file is not None:
        model.load_weights(weights_file)

    # Prepare loss and optimizer.
    # TODO: add learing rate decay.
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=1e-4))

    # Prepare location for saved checkpoints.
    checkpoint_path = 'checkpoint/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(checkpoint_path + 'weights.{epoch:03d}-{val_loss:.2f}.hdf5',
                                          save_weights_only=True)

    train_generator = DIV2K_RAM_generate(train=True, batch_size=BATCH_SIZE, crop_size=32)
    valid_generator = DIV2K_RAM_generate(train=False, batch_size=1, crop_size=None)

    # With default settings on 1 GPU, this should take roughly 1 day to train.
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=1000,
                        epochs=N_EPOCHS,
                        callbacks=[checkpoint_callback],
                        validation_data=valid_generator,
                        validation_steps=100)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_weights', type=str,
                        help='The path to the saved model weights to initialize training from, if any.')

    args = parser.parse_args()

    train(args.model_weights)
