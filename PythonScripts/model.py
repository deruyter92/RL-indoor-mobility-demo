import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, imsize, in_channels, out_channels):
        super(DQN, self).__init__()
        # Convulutional input layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Fully connected output layer
        imsize = imsize if type(imsize) is tuple else (imsize,imsize)
        n_hidden = ((imsize[0]-21)//8)*((imsize[1]-21)//8) * 32 
        self.head = nn.Linear(n_hidden, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))    