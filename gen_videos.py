import cv2

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
from model import SRGAN_g
from timeit import default_timer as timer

class SRGAN:
    def __init__(self):
        # Form the generator network.
        self.t_image = tf.placeholder('float32', [None, None, None, 3], name='input_image')
        self.net_g = SRGAN_g(self.t_image, is_train=False, reuse=False)

        # Restore the generator weights.
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(self.sess)
        tl.files.load_and_assign_npz(sess=self.sess, name='checkpoint/g_srgan.npz', network=self.net_g)

    def process_image(self, lr_frame):
        # Do superresolution.
        lr_frame = (lr_frame / 127.5) - 1
        hr_frame = self.sess.run(self.net_g.outputs, {self.t_image: [lr_frame]})
        hr_frame = (hr_frame + 1) * 127.5
        hr_frame = np.uint8(hr_frame + 0.5)
        return hr_frame[0]

class Bicubic:
    def __init__(self, scale):
        self.scale = scale

    def process_image(self, lr_frame):
        h_scaled = int(lr_frame.shape[0] * self.scale)
        w_scaled = int(lr_frame.shape[1] * self.scale)
        hr_frame = scipy.misc.imresize(lr_frame, [h_scaled, w_scaled], interp='bicubic', mode=None)
        return hr_frame

class VideoFrames:
    def __init__(self, video_path):
        self.vidcap = cv2.VideoCapture(video_path)
        if not self.vidcap.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
                           "trying to open a webcam, make sure your video_path is an integer!"))

        self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        return self

    def __next__(self):
        flag, frame = self.vidcap.read()
        if not flag:
            print("Finished reading video!")
            self.vidcap.release()
            raise StopIteration
        return frame

def process_video(video_path, lr_method, sr_methods, frame_skip=1):
    # Open a video for reading
    video_reader = VideoFrames(video_path)
    print("Ground Truth resolution = {} x {}".format(video_reader.width, video_reader.height))

    for frame_count, lr_frame in enumerate(video_reader):
        if frame_count % frame_skip == 0:
            if lr_method is not None:
                lr_frame = lr_method.process_image(lr_frame)
            sr_frames = []
            for sr_method in sr_methods:
                sr_frame = sr_method.process_image(lr_frame)
                sr_frames.append(sr_frame)

            sr_joined = np.concatenate(sr_frames, axis=1)
            cv2.imshow('frame', sr_joined)
            cv2.waitKey(1)

            if frame_count == 0:
                # Define the codec and create VideoWriter objects
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                write_shape = (sr_joined.shape[1], sr_joined.shape[0])
                print("Output resolution = {}".format(write_shape))
                video_writer = cv2.VideoWriter('samples/evaluate/joined.avi', fourcc, 20.0, write_shape)

            print("Writing frame {}".format(frame_count))
            video_writer.write(sr_joined)

    video_writer.release()

def process_videos(video_paths):
    lr_method = Bicubic(0.25)
    sr_methods = [Bicubic(4), SRGAN()]

    for path in video_paths:
        process_video(path, lr_method, sr_methods, frame_skip=2)

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
    #join_videos()
    process_videos([args.video])

    delta_time = timer() - start_time
    print("took: %4.4fs" % delta_time)
