import torchvision.models as models
import torch.nn as nn

model = models.resnet101(pretrained=True)
HighFeatures = nn.Sequential(*(list(model.children())[:-2]))
LowFeatures = nn.Sequential(*(list(model.children())[0:6]))

print(HighFeatures)

#HighFeatures.forward()

#print('test', LowFeatures)

#featuresHigh = ModelResHigh('Input')
#featuresLow = ModelResLow('Input')