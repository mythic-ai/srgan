#! /usr/bin/python
# -*- coding: utf8 -*-

from imageio import imread
from timeit import default_timer as timer

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from model import get_model
from preprocessing import DIV2K_generate

###====================== HYPER-PARAMETERS ===========================###
BATCH_SIZE = 16
N_EPOCHS = 1000

def train():
    model = get_model()

    # Prepare loss and optimizer.
    # TODO: add learing rate decay.
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=1e-4))

    # Prepare location for saved checkpoints.
    checkpoint_callback = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                          save_weights_only=True)

    # The original paper runs for 200 epochs of 1000 batches each.
    # We instead break it into 1000 epochs of 200 batches each.
    # The authors report that this should take 1 day on a Titan XP.
    model.fit_generator(generator=DIV2K_generate(train=True, batch_size=BATCH_SIZE),
                        steps_per_epoch=3200/BATCH_SIZE,
                        epochs=N_EPOCHS,
                        callbacks=[checkpoint_callback],
                        validation_data=DIV2K_generate(train=False, batch_size=BATCH_SIZE),
                        validation_steps=96/BATCH_SIZE)

def evaluate():
    # Load model
    model = get_model()
    model.load_weights('checkpoints/weights.3-3.33.hdf5')

    # Read image
    generator = DIV2K_generate(train=False, batch_size=1)
    lr_img = generator.next()

    # Generate in high resolution
    hr_img = model.predict(lr_img)

    # TODO: save image to a file or display it

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')

    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknown --mode")
