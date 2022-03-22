import torch
import torch.nn as nn
import torchvision.models as models
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = models.wide_resnet101_2(pretrained=False, progress=True)
        # self.resnet = models.wide_resnet50_2(pretrained=False, progress=True)
        # self.resnet = models.resnet101(pretrained=False, progress=True)
        # self.vgg = models.vgg16_bn(pretrained=False, progress=True)
        # self.layer = nn.Linear(1000, 11)
        
    def forward(self, x):
        x = self.resnet(x)
        # x = self.layer(x)
        return x