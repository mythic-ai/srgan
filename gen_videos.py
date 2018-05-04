import cv2

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
from model import SRGAN_g
from timeit import default_timer as timer

class SRGAN:
    """An object that performs 4x super-resolution using SRGAN"""

    def __init__(self):
        """Initializes the SRGAN generator network using pre-trained weights"""

        # Form the generator network.
        self.t_image = tf.placeholder('float32', [None, None, None, 3], name='input_image')
        self.net_g = SRGAN_g(self.t_image, is_train=False, reuse=False)

        # Restore the generator weights.
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(self.sess)
        tl.files.load_and_assign_npz(sess=self.sess, name='checkpoint/g_srgan.npz', network=self.net_g)

    def process_image(self, lr_frame):
        """Returns the super-resolved version of lr_frame"""
        lr_frame = (lr_frame / 127.5) - 1
        hr_frame = self.sess.run(self.net_g.outputs, {self.t_image: [lr_frame]})
        hr_frame = (hr_frame + 1) * 127.5
        hr_frame = np.uint8(hr_frame + 0.5)
        return hr_frame[0]

class Bicubic:
    """An object that performs bicubic up- or down-sampling at a given scale"""

    def __init__(self, scale):
        """Sets the scale factor for up-sampling (if greater than 1) or down-sampling (if less than 1)"""
        self.scale = scale

    def process_image(self, lr_frame):
        """Returns an image whose dimensions are multiplied by scale, using bicubic interpolation"""
        h_scaled = int(lr_frame.shape[0] * self.scale)
        w_scaled = int(lr_frame.shape[1] * self.scale)
        hr_frame = scipy.misc.imresize(lr_frame, [h_scaled, w_scaled], interp='bicubic', mode=None)
        return hr_frame

class VideoFrames:
    """An iterator that reads a sequence of frames from a video file"""

    def __init__(self, video_path):
        """Initializes the iterator from a video file"""
        print("Reading from {}...".format(video_path))
        self.vidcap = cv2.VideoCapture(video_path)
        if not self.vidcap.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
                           "trying to open a webcam, make sure your video_path is an integer!"))

        self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(" Input resolution = {}".format((self.width, self.height)))

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next frame, or StopIteration if there are no more frames"""
        flag, frame = self.vidcap.read()
        if not flag:
            print("Finished reading video!")
            self.vidcap.release()
            raise StopIteration
        return frame

def process_video(filename, lr_method, sr_methods, frame_skip=1):
    # Open a video for reading
    start_time = timer()

    in_path = 'data/' + filename + '.y4m'
    video_reader = VideoFrames(in_path)

    for frame_count, lr_frame in enumerate(video_reader):
        if frame_count % frame_skip == 0:
            if lr_method is not None:
                lr_frame = lr_method.process_image(lr_frame)
            sr_frames = []
            for sr_method in sr_methods:
                sr_frame = sr_method.process_image(lr_frame)
                sr_frames.append(sr_frame)
            sr_joined = np.concatenate(sr_frames, axis=1)

            # These lines must be commented when running via SSH
            #cv2.imshow('frame', sr_joined)
            #cv2.waitKey(1)

            if frame_count == 0:
                write_shape = (sr_joined.shape[1], sr_joined.shape[0])
                print("Output resolution = {}".format(write_shape))

                # Define the codec and create VideoWriter object
                out_path = 'samples/evaluate/' + filename + '.avi'
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(out_path, fourcc, 20.0, write_shape)

            print("Writing frame {}".format(frame_count))
            video_writer.write(sr_joined)

    video_writer.release()

    delta_time = timer() - start_time
    print("Processed {} in {:.4f}s".format(in_path, delta_time))

def process_videos():
    lr_method = Bicubic(0.25)
    sr_methods = [Bicubic(4), SRGAN()]

    filenames = ['football_cif',
                 'aspen_1080p',
                 'blue_sky_1080p25',
                 'controlled_burn_1080p',
                 'crowd_run_2160p50',
                 'dinner_1080p30',
                 'ducks_take_off_2160p50',
                 'factory_1080p30',
                 'FourPeople_1280x720_60',
                 'in_to_tree_2160p50']
    for filename in filenames:
        process_video(filename, lr_method, sr_methods, frame_skip=1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Unused parameter
    parser.add_argument('--video', type=str, default='data/football_cif.y4m',
                        help='The optional path to a .mp4 file to run the SSD model on, frame by frame.'
                            'If this parameter is unspecified, the program will use the video stream from the webcam.')

    args = parser.parse_args()

    start_time = timer()
    process_videos()

    delta_time = timer() - start_time
    print("Total time: {:.4f}s".format(delta_time))
