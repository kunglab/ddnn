from __future__ import print_function
import torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

import torch.nn as nn

def _layer(in_channels, out_channels, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResLayer, self).__init__()
        self.c1 = _layer(in_channels, out_channels)
        self.c2 = _layer(out_channels, out_channels, activation=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        
        # residual connection
        if x.shape[1] == h.shape[1]:
            h += x

        h = self.activation(h)

        return h


class DeviceModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeviceModel, self).__init__()
        self.model = nn.Sequential(
            ResLayer(in_channels, 8),
            ResLayer(8, 8),
            ResLayer(8, 16),
            ResLayer(16, 16),
            ResLayer(16, 16),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, out_channels)
    
    def forward(self, x):
        B = x.shape[0]
        h = self.model(x)
        p = self.pool(h)
        return h, self.classifier(p.view(B, -1))
    

class DDNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_devices):
        super(DDNN, self).__init__()
        self.num_devices = num_devices
        self.device_models = []
        for _ in range(num_devices):
            self.device_models.append(DeviceModel(in_channels, out_channels))
        self.device_models = nn.ModuleList(self.device_models)

        cloud_input_channels = 16*num_devices
        self.cloud_model = nn.Sequential(
            ResLayer(cloud_input_channels, 64),
            ResLayer(64, 64),
            ResLayer(64, 128),
            nn.AvgPool2d(2, 2),
            ResLayer(128, 128),
            ResLayer(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, out_channels)

    def forward(self, x):
        B = x.shape[0]
        hs, predictions = [], []
        for i, device_model in enumerate(self.device_models):
            h, prediction = device_model(x[:, i])
            hs.append(h)
            predictions.append(prediction)

        h = torch.cat(hs, dim=1)

        # dropout half of devices (training)
        if self.training:
            idxs = torch.randperm(self.num_devices) 
            if torch.cuda.is_available:
                idxs = idxs.cuda()
            h[:, idxs][:self.num_devices//2] = 0

        h = self.cloud_model(h)
        h = self.pool(h)
        prediction = self.classifier(h.view(B, -1))
        predictions.append(prediction)
        return predictions