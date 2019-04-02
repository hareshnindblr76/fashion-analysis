import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
import numpy as np
pretrained_nets = {'resnet18':models.resnet18, 'resnet34':models.resnet34, 'resnet50':models.resnet50}
reduced_channels=128
conv3_out = {'resnet18':256,'resnet50':1024}
class FashioNet(nn.Module):
    def __init__(self, backbone, num_classes, train_blocks=None):
        super(FashioNet,self).__init__()
        self.model = pretrained_nets[backbone](pretrained=True)
        self.backbone = nn.Sequential(*list(self.model.children())[:-3] )
        if train_blocks:
            self._freeze_params(train_blocks)
        self.strided_conv = nn.Conv2d(conv3_out[backbone],reduced_channels,(3,3),2)
        self.leakyRelu = nn.LeakyReLU()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        #print(self.backbone)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(reduced_channels,num_classes)
    def _freeze_params(self, train_blocks):
        for name, layer in self.model.named_children():
            if name not in train_blocks:
                #print("freezing layer ",layer)
                for param in layer.parameters():
                    param.requires_grad=False
            else:
                #print("unfreezing layer",name)
                for param in layer.parameters():
                    param.requires_grad = True
    def forward(self,x):

        x = self.backbone(x)
        x = self.strided_conv(x)
        x = self.leakyRelu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #print(x.shape)
        return x

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}