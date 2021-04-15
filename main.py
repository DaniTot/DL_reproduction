import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch
import numpy as np

import os
import cv2
import numpy as np
import time

from collections import Counter
import tracemalloc

from FeatureExtractor import FeatureExtractor
from LSTS import Agg_LSTS, Com_LSTS
from MemProfile import display_top


class Train:

    def __init__(self, batch_size, name):
        if name == "Daniel":
            ilscvrc_path = os.path.join(os.path.abspath(os.sep), 'Users', 'tothd', 'Documents', 'TU Delft', 'Msc',
                                    'Deep Learning')
        elif name == "Ruben":
            ilscvrc_path = os.path.join(os.path.abspath(os.sep), 'Users', 'ruben', 'Downloads')
        else:
            print("Not a correct name")

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
        self.aspect_ratio = [1280, 720]

        self.FeEx = FeatureExtractor()
        self.LSTS = Com_LSTS([1, 1024, 18, 32])

        self.times = [time.time()]
        self.time = time.time()

    def batch_randomizer(self):
        # count how many datasets we have
        mp4_count = 0
        for f in os.listdir(self.train_set_path):
            if os.path.isdir(os.path.join(self.train_set_path, f)):
                mp4_count += 1

        self.batch_data_idx = np.random.randint(0, mp4_count, self.batch_size)

    def select_vid(self):
        while True:
            directory = os.fsencode(self.train_set_path)
            mp4_lst = os.listdir(directory)
            self.mp4_path = os.path.join(self.train_set_path,
                                         os.fsdecode(mp4_lst[self.batch_data_idx[self.iteration_count]]))

            directory = os.fsencode(self.mp4_path)
            frame_lst = os.listdir(directory)
            frame_path = os.path.join(self.mp4_path, os.fsdecode(frame_lst[0]))
            frame = cv2.imread(frame_path)
            if frame.shape[0] == self.aspect_ratio[1] and frame.shape[1] == self.aspect_ratio[0]:
                # print("aspects", frame.shape, self.aspect_ratio)
                break
            else:
                # print("aspects", frame.shape, self.aspect_ratio)
                add_count = 1
                while self.batch_data_idx[self.iteration_count] + add_count in self.batch_data_idx:
                    add_count += 1

                self.batch_data_idx[self.iteration_count] += add_count
        return

    def load_frame(self):
        directory = os.fsencode(self.mp4_path)
        frame_lst = os.listdir(directory)
        frame_path = os.path.join(self.mp4_path, os.fsdecode(frame_lst[self.frame_count]))
        print(frame_path)
        if os.path.isfile(frame_path):
            # new_frame = Image.open(frame_path)
            frame = cv2.imread(frame_path)
            self.new_frame, self.new_scale = self.resize(frame, self.SCALES[0], self.SCALES[1], stride=0)
            # print("resized", self.new_frame.shape, self.new_scale)
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
                    print("Randomize batch, Select vid, Load frame--- %s seconds ---" % (time.time() - self.time))
                    self.time = time.time()
                    # Get thefeature vectors from the new frame
                    # SRFU segment
                    if self.iteration_count % self.key_frame_interval == 0:
                        key_frame = True
                        conv_feat_oldkey, conv_feat_newkey, feature_low_level = self.FeEx.get_feature(self.new_frame,
                                                                                                      key_frame)
                        print("Extract features--- %s seconds ---" % (time.time() - self.time))
                        self.time = time.time()
                        # snapshot = tracemalloc.take_snapshot()
                        # display_top(snapshot)
                        print()
                        # print("shapes", conv_feat_oldkey.shape, conv_feat_newkey.shape, feature_low_level.shape)
                        # print(conv_feat_oldkey.shape, conv_feat_newkey.shape, feature_low_level.shape)
                        ret = self.LSTS.get_input(conv_feat_oldkey, conv_feat_newkey)
                        if ret == 0:
                            print(self.LSTS.F_0_embedded.shape, self.LSTS.F_0.shape, self.FeEx.frame.shape)
                            print(self.LSTS.F_1_embedded.shape, self.LSTS.F_1.shape, self.FeEx.frame.shape)
                            assert False
                        else:
                            # self.LSTS.WeightGenerate()
                            # self.LSTS.Aggregate()
                            self.LSTS.DoImage(weight=True, aggregate=True, gradients=True, update=False)
                            print("Wights+Aggregate--- %s seconds ---" % (time.time() - self.time))
                            self.time = time.time()
                            # snapshot = tracemalloc.take_snapshot()
                            # display_top(snapshot)
                            print()
                            assert False
                            F_out = self.LSTS.quality_network(self.LSTS.F_1, self.LSTS.F_pred, key_frame)
                            self.LSTS.DoImage(weight=False, aggregate=False, gradients=False, update=True)

                        # feat_task = mx.sym.take(mx.sym.Concat(*[feat_task, conv_feat_oldkey], dim=0), eq_flag_key2key)

                    # DFA segment
                    else:
                        key_frame = False
                        conv_feat_oldkey, conv_feat_newkey, feature_low_level = self.FeEx.get_feature(self.new_frame,
                                                                                                      key_frame)
                        high_feat_current = self.LSTS.low2high_transform(feature_low_level)
                        self.LSTS.get_input(self.LSTS.F_mem, high_feat_current)
                        self.LSTS.DoImage(weight=True, aggregate=True, gradients=True, update=False)
                        F_out = self.LSTS.quality_network(self.LSTS.F_1, self.LSTS.F_pred, key_frame)
                        self.LSTS.DoImage(weight=False, aggregate=False, gradients=False, update=True)

                    # TODO: RPN
                    # TODO: ROI proposal

                    return
                else:
                    break

# tracemalloc.start()

Tr = Train(1, "Daniel")
Tr.get_feature()