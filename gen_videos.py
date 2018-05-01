import cv2

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
from model import SRGAN_g
from timeit import default_timer as timer

def shrink_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise IOError(("Couldn't open video file or webcam. If you're "
                       "trying to open a webcam, make sure your video_path is an integer!"))

    scale = 4
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) / scale)
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) / scale)

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_lr = cv2.VideoWriter('samples/evaluate/lr.avi', fourcc, 20.0, (width, height))

    while True:
        flag, frame = vidcap.read()
        if not flag:
            print("Done!")
            return

        lr_frame = scipy.misc.imresize(frame, [height, width], interp='bicubic', mode=None)
        print(lr_frame.shape)
        vid_lr.write(lr_frame)

    vidcap.release()
    vid_lr.release()

def video_superresolution(video_path):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise IOError(("Couldn't open video file or webcam. If you're "
                       "trying to open a webcam, make sure your video_path is an integer!"))

    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 4

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidbic = cv2.VideoWriter('samples/evaluate/bic.avi', fourcc, 20.0, (width * scale, height * scale))
    vidgen = cv2.VideoWriter('samples/evaluate/gen.avi', fourcc, 20.0, (width * scale, height * scale))

    # Form the generator network.
    t_image = tf.placeholder('float32', [None, None, None, 3], name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    # Restore the generator weights.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    checkpoint_dir = "checkpoint"
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    i = 0
    while True:
        i += 1
        flag, lr_frame = vidcap.read()
        if not flag:
            print("Done!")
            return

        bic_frame = scipy.misc.imresize(lr_frame, [height * scale, width * scale], interp='bicubic', mode=None)
        vidbic.write(bic_frame)
        print("frame {}".format(i))

        if i % 1 == 0:
            # Do superresolution.
            lr_frame = (lr_frame / 127.5) - 1
            hr_frame = sess.run(net_g.outputs, {t_image: [lr_frame]})
            hr_frame = (hr_frame + 1) * 127.5
            hr_frame = np.uint8(hr_frame + 0.5)

            print(hr_frame.shape)
            print(hr_frame.min())
            print(hr_frame.max())

            cv2.imshow('frame', hr_frame[0])
            cv2.waitKey(1)
            vidgen.write(hr_frame[0])

    vidcap.release()
    vidbic.release()
    vidgen.release()
    cv2.destroyAllWindows()

def join_videos():
    vidbic = cv2.VideoCapture('samples/evaluate/bic.avi')
    vidgen = cv2.VideoCapture('samples/evaluate/gen.avi')
    if not vidbic.isOpened() or not vidgen.isOpened():
        raise IOError(("Couldn't open video file or webcam. If you're "
                       "trying to open a webcam, make sure your video_path is an integer!"))

    width = int(vidbic.get(cv2.CAP_PROP_FRAME_WIDTH) + vidbic.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidbic.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidjoined = cv2.VideoWriter('samples/evaluate/joined.avi', fourcc, 20.0, (width, height))

    while True:
        flag1, frame1 = vidbic.read()
        flag2, frame2 = vidgen.read()
        if not flag1 or not flag2:
            print("Done!")
            return

        joined_frame = np.concatenate((frame1, frame2), axis=1)
        print(joined_frame.shape)
        vidjoined.write(joined_frame)

    vidbic.release()
    vidgen.release()
    vidjoined.release()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    start_time = timer()

    parser.add_argument('--video', type=str, default='data2017/football_cif.y4m',
                        help='The optional path to a .mp4 file to run the SSD model on, frame by frame.'
                            'If this parameter is unspecified, the program will use the video stream from the webcam.')

    args = parser.parse_args()

    #shrink_video('data2017/Netflix_Aerial_4096x2160_60fps_10bit_420.y4m')
    #video_superresolution(args.video)
    join_videos()

    delta_time = timer() - start_time
    print("took: %4.4fs" % delta_time)
