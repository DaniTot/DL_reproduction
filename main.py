import os
import cv2
import numpy as np

from FeatureExtractor import FeatureExtractor
from LSTS import Agg_LSTS, Com_LSTS


class Train:
    def __init__(self, batch_size):
        ilscvrc_path = os.path.join(os.path.abspath(os.sep), 'Users', 'tothd', 'Documents', 'TU Delft', 'Msc', 'Deep Learning')
        vid_path = os.path.join(ilscvrc_path, 'ILSVRC2015', 'Data', 'VID')
        self.train_set_path = os.path.join(vid_path, 'train', 'ILSVRC2015_VID_train_0001')
        assert os.path.isdir(self.train_set_path), self.train_set_path

        self.mp4_path = None

        self.new_frame = None
        self.new_scale = None

        self.batch_data_idx = None

        self.batch_size = batch_size
        self.iteration_count = 0

        self.frame_count = 0
        self.key_frame_interval = 10

        self.SCALES = [600, 1000]  # first is scale (the shorter side); second is max size

        self.FeEx = FeatureExtractor()

    def batch_randomizer(self):
        # count how many datasets we have
        mp4_count = 0
        for f in os.listdir(self.train_set_path):
            if os.path.isdir(os.path.join(self.train_set_path, f)):
                mp4_count += 1

        self.batch_data_idx = np.random.randint(0, mp4_count, self.batch_size)

    def select_vid(self):
        directory = os.fsencode(self.train_set_path)
        mp4_lst = os.listdir(directory)
        self.mp4_path = os.path.join(self.train_set_path,
                                     os.fsdecode(mp4_lst[self.batch_data_idx[self.iteration_count]]))
        # print(self.mp4_path)
        return

    def load_frame(self):
        directory = os.fsencode(self.mp4_path)
        frame_lst = os.listdir(directory)
        frame_path = os.path.join(self.mp4_path, os.fsdecode(frame_lst[self.frame_count]))

        if os.path.isfile(frame_path):
            # new_frame = Image.open(frame_path)
            frame = cv2.imread(frame_path)
            self.new_frame, self.new_scale = self.resize(frame, self.SCALES[0], self.SCALES[1], stride=0)

            self.frame_count += 1
            return True

        else:
            self.frame_count = 0
            self.iteration_count += 1
            return False

    @staticmethod
    def resize(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
        """
        only resize input image to target size and return scale
        :param im: BGR image input by opencv
        :param target_size: one dimensional size (the short side)
        :param max_size: one dimensional max size (the long side)
        :param stride: if given, pad the image to designated stride
        :param interpolation: if given, using given interpolation method to resize image
        :return:
        """
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

        if stride == 0:
            return im, im_scale
        else:
            # pad to product of stride
            im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
            im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
            im_channel = im.shape[2]
            padded_im = np.zeros((im_height, im_width, im_channel))
            padded_im[:im.shape[0], :im.shape[1], :] = im
            return padded_im, im_scale

    def get_feature(self):
        # Setup
        self.batch_randomizer()
        for iteration in range(self.batch_size):
            self.select_vid()
            while True:
                if self.load_frame():

                    # Get thefeature vectors from the new frame
                    if self.iteration_count % self.key_frame_interval == 0:
                        key_frame = True
                    else:
                        key_frame = False

                    conv_feat_oldkey, conv_feat_newkey, feature_low_level = self.FeEx.get_feature(self.new_frame, key_frame)

                    # TODO: align conv_feat_oldkey to conv_feat_newkey

                    # TODO:

                    #TODO: key2key quality network

                    #TODO: etc...

                    return
                else:
                    break
