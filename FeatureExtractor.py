import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch
import numpy as np
import cv2

import os


class FeatureExtractor:

    def __init__(self):

        self.frame = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

        self.model = None
        self.HighFeatures = None
        self.LowFeatures = None

        self.new_frame = None
        self.key_frame = None

        self.model_setup()

    def model_setup(self):
        self.model = models.resnet101(pretrained=True)
        self.HighFeatures = nn.Sequential(*(list(self.model.children())[:-2]))
        self.LowFeatures = nn.Sequential(*(list(self.model.children())[0:6]))
        print(self.model)
        self.HighFeatures.eval()
        self.LowFeatures.eval()

    def frame_setup(self):

        if self.frame is None:
            # Very first frame in the video/batch

            img_old = self.transform(self.new_frame)
            img_new = self.transform(self.new_frame)
            img_nonkey = self.transform(self.new_frame)

        else:
            img_old, img_new, img_nonkey = torch.split(self.frame, 1, dim=0)
            if self.key_frame is True:
                # This is a key frame, so preserve the previous key and non-key frame
                # Add the new frame
                img_old = img_new
                img_new = self.transform(self.new_frame)
            else:
                img_nonkey = self.transform(self.new_frame)

        self.frame = torch.cat((torch.unsqueeze(img_old, 0),
                                torch.unsqueeze(img_new, 0),
                                torch.unsqueeze(img_nonkey, 0)), 0)

        return self.frame

    def new_vid(self):
        self.frame = None

    def get_feature(self, new_frame, key_frame):
        ###
        # lsts_rfcn/symbols/lsts_network_gaussian.py -> get_train_symbol_impression()
        ###
        self.new_frame = new_frame
        self.key_frame = key_frame
        self.frame_setup()

        # HighFeatures takes old_key and new_key
        high = self.HighFeatures(torch.cat(torch.split(self.frame, 1)[:2], 0))
        conv_feat_oldkey, conv_feat_newkey = torch.split(high, 1)

        # LowFeatures takes old_key, new_key, data_cur
        feature_low_level = self.LowFeatures(torch.split(self.frame, 1)[2])

        # Align old and new key features
        # offset = mx.sym.Variable(name='init_offset', shape=(N, 2), lr_mult=10000*cfg.TRAIN.lr)

        return conv_feat_oldkey, conv_feat_newkey, feature_low_level
