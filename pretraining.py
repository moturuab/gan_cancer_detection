import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from collections import OrderedDict

from MRIDataset import MRIDataset

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--load_g", type=str, default=None, help="generator model to load")
parser.add_argument("--load_d", type=str, default=None, help="discriminator model to load")

opt = parser.parse_args()
print(opt)

class ConvAutoEncoder(nn.Module):
    def __init__(self, feature_size):
        super(ConvAutoEncoder, self).__init__()
        self.feature_size = feature_size
        #Encoder: 1 -> 16 -> 4 ->
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.feature_size, kernel_size=3, stride=2, padding=(1, 1, 1)),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            torch.nn.ReLU()
        )  # (6, 362, 128)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.feature_size, self.feature_size * 2, kernel_size=3, stride=2, padding=(1, 1, 1)),
            # torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            torch.nn.ReLU()
        )  # (3, 181, 64)

        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=(1, 1, 1)),
        #     torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
        #     torch.nn.ReLU()
        # ) # (1, 181, 64)
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.feature_size * 2, self.feature_size * 4, kernel_size=3, stride=(3, 2, 1), padding=(0, 0, 1)),
            torch.nn.MaxPool3d(kernel_size=(1, 2, 2)),
            torch.nn.ReLU()
        )  # (1, 45, 32)
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.feature_size * 4, self.feature_size * 8, kernel_size=3, stride=2, padding=(1, 1, 1)),
            torch.nn.ReLU()
        )  # (1, 23, 16)

        self.flatten = Flatten()

        #Decoder

        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size * 4, self.feature_size * 8, kernel_size=3, stride=2, padding=(1, 1, 1)),
            torch.nn.ReLU()
        )  # (1, 23, 16)

        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size * 2, self.feature_size * 4, kernel_size=3, stride=(3, 2, 1), padding=(0, 0, 1)),
            torch.nn.ReLU()
        )  # (1, 45, 32)

        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size, self.feature_size * 2, kernel_size=3, stride=2, padding=(1, 1, 1)),
            # torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            torch.nn.ReLU()
        )  # (3, 181, 64)

        self.layer9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1, self.feature_size, kernel_size=3, stride=2, padding=(1, 1, 1)),
            torch.nn.ReLU()
        ) 
    
    def forward(self, volume):
        #Encoder
        out = self.layer1(volume)
        out = self.layer2(volume)
        out = self.layer4(volume)
        out = self.layer5(volume)

        # Latent Representation
        out = self.flatten(out)

        #Decoder
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        return out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

model = ConvAutoEncoder(16)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

dataloader = torch.utils.data.DataLoader(
    MRIDataset(
        csv_file="annotations.csv",
        root_dir="wbmri",
        # transform=transforms.Compose(
        #     [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        # ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

n_epochs = 30

for epoch in range (1, n_epochs+1):
    train_loss = 0.0

    for i, volume in enumerate(dataloader):

        optimizer.zero_grad()

        outputs = model(volume)

        loss = criterion(outputs, volume)
        loss.backward()
        optimizer.step()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [Loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader),loss.item())
        )

        batches_done = epoch * len(dataloader) + i

        if batches_done % opt.sample_interval == 0:
            im = outputs.cpu().detach().numpy()[0, 0, 11, :, :]
            np.save(im, "epoch_{}_batch_{}.png".format(epoch, i))
            plt.imshow(im, cmap="gray")
            plt.draw()
            plt.pause(0.001)