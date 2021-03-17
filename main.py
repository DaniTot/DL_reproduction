import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch

import os
import cv2


model = models.resnet101(pretrained=True)
HighFeatures = nn.Sequential(*(list(model.children())[:-2]))
LowFeatures = nn.Sequential(*(list(model.children())[0:6]))
HighFeatures.eval()
LowFeatures.eval()

print(HighFeatures)
print(LowFeatures)


transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


vid_path = 'C:\\Users\\tothd\\Documents\\TU Delft\\Msc\\Deep Learning\\ILSCVRC2015\\Data\\VID\\snippets'
train_set_1_path = os.path.join(vid_path, 'train\\ILSVRC2015_VID_train_0000')


directory = os.fsencode(train_set_1_path)

for file in os.listdir(directory):
    mp4_path = os.path.join(train_set_1_path, os.fsdecode(file))

vidcap = cv2.VideoCapture(mp4_path)
success = True
while success:
    success, image = vidcap.read()
    cv2.imwrite("frame.jpg", image)  # save frame as JPEG file

    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)
    features_high = HighFeatures(batch_t)
    features_low = LowFeatures(batch_t)
    print(features_high.shape)
    print(features_low.shape)

