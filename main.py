import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch
import numpy as np

import os
from PIL import Image


class Train:

    def __init__(self, batch_size):
        ilscvrc_path = os.path.join(os.path.abspath(os.sep), 'Users', 'tothd', 'Documents', 'TU Delft', 'Msc',
                                    'Deep Learning')
        vid_path = os.path.join(ilscvrc_path, 'ILSVRC2015', 'Data', 'VID')
        self.train_set_path = os.path.join(vid_path, 'train', 'ILSVRC2015_VID_train_0001')
        assert os.path.isdir(self.train_set_path), self.train_set_path

        self.mp4_path = None

        self.frame = None
        self.data_cur = None

        self.batch_data_idx = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

        self.model = None
        self.HighFeatures = None
        self.LowFeatures = None

        self.batch_size = batch_size
        self.iteration_count = 0

        self.frame_count = 0

    def model_setup(self):
        self.model = models.resnet101(pretrained=True)
        self.HighFeatures = nn.Sequential(*(list(self.model.children())[:-2]))
        self.LowFeatures = nn.Sequential(*(list(self.model.children())[0:6]))
        self.HighFeatures.eval()
        self.LowFeatures.eval()

    def batch_randomizer(self):
        # count how many datasets we have
        mp4_count = 0
        for f in os.listdir(self.train_set_path):
            if os.path.isdir(os.path.join(self.train_set_path, f)):
                mp4_count += 1

        self.batch_data_idx = np.random.randint(0, mp4_count, self.batch_size)

    def load_frame(self):
        directory = os.fsencode(self.mp4_path)
        frame_lst = os.listdir(directory)
        frame_path = os.path.join(self.mp4_path, os.fsdecode(frame_lst[self.frame_count]))

        if os.path.isfile(frame_path):
            if self.frame is None:
                # Get the first two frames

                img_new = self.transform(Image.open(frame_path))
                self.frame = torch.unsqueeze(img_new, 0)
                self.frame_count += 1

                second_frame_path = os.path.join(self.mp4_path, os.fsdecode(frame_lst[self.frame_count]))
                second_img_new = self.transform(Image.open(second_frame_path))

                self.frame = torch.cat((self.frame, torch.unsqueeze(second_img_new, 0)), 0)

            else:
                # Preserve the previous frame
                self.frame = torch.split(self.frame, 2, dim=0)[1]

                # Add the new frame
                img_new = self.transform(Image.open(frame_path))
                self.frame = torch.cat((self.frame, torch.unsqueeze(img_new, 0)), 0)

            self.frame_count += 1
            return True
        else:
            self.frame_count = 0
            self.frame = None
            self.iteration_count += 1
            return False

    def select_vid(self):
        directory = os.fsencode(self.train_set_path)
        mp4_lst = os.listdir(directory)
        self.mp4_path = os.path.join(self.train_set_path, os.fsdecode(mp4_lst[self.batch_data_idx[self.iteration_count]]))

        return

    def run(self):
        # Setup
        self.model_setup()
        self.batch_randomizer()
        for iteration in range(self.batch_size):
            self.select_vid()
            while True:
                if self.load_frame():
                    # The image is loaded here

                    ###
                    # lsts_rfcn/symbols/lsts_network_gaussian.py -> get_train_symbol_impression()
                    ###

                    # HighFeatures takes old_key and new_key
                    high = self.HighFeatures(self.frame)
                    features = torch.split(high, 1)

                    conv_feat_oldkey = features[0]
                    conv_feat_newkey = features[1]

                    print(conv_feat_newkey)

                    # LowFeatures takes old_key, new_key, data_cur
                    # TODO: Get data_cur somehow
                    low = self.LowFeatures(torch.cat((self.frame, self.data_cur), 0))
                    features_low_level_slice = torch.split(low, 1)


                    #TODO: Align conv_feat_oldkey and conv_feat_newkey

                    #TODO: key2key quality network

                    #TODO: etc...

                    return
                else:
                    break


Tr = Train(1)
Tr.run()
