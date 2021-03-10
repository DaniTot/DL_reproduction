import torchvision.models as models
import torch.nn as nn

resnet101 = models.resnet101(True, True)
print(resnet101)
ModelResHigh = nn.Sequential(*(list(resnet101.children())[:]))
ModelResLow = nn.Sequential(*(list(resnet101.children())[:-2])) #2 isn't correct, look at questions.

#featuresHigh = ModelResHigh('Input')
#featuresLow = ModelResLow('Input')