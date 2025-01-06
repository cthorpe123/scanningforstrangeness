import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)

def maxpool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def dropout(prob):
    return nn.Dropout(prob)

def reinit_layer(layer, leak=0.0, use_kaiming_normal=True):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        if use_kaiming_normal:
            nn.init.kaiming_normal_(layer.weight, a = leak)
        else:
            nn.init.kaiming_uniform_(layer.weight, a = leak)
            layer.bias.data.zero_()

class Sigmoid(nn.Module):
    def __init__(self, out_range=None):
        super(Sigmoid, self).__init__()
        if out_range is not None:
            self.low, self.high = out_range
            self.range = self.high - self.low
        else:
            self.low = None
            self.high = None
            self.range = None

    def forward(self, x):
        if self.low is not None:
            return torch.sigmoid(x) * (self.range) + self.low
        else:
            return torch.sigmoid(x)

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3):
        super(ConvBlock, self).__init__()
        k_pad = int((k_size - 1)/2)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.identity = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=1)
        reinit_layer(self.conv1)
        reinit_layer(self.conv2)

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.relu(x + identity)

class TransposeConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3):
        super(TransposeConvBlock, self).__init__()
        k_pad = int((k_size - 1)/2)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=k_size, padding=k_pad, output_padding=1, stride=2),
            nn.GroupNorm(8, c_out),
            nn.ReLU(inplace=True)
        )
        reinit_layer(self.block[0])

    def forward(self, x):
        return self.block(x)
