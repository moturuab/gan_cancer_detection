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
from torch.nn.utils import spectral_norm

import torch

from collections import OrderedDict

from MRIDataset import MRIDataset

import matplotlib.pyplot as plt
import matplotlib.image

import time


from scipy import misc
from imageio import imwrite

os.makedirs("g_z", exist_ok=True)
os.makedirs("networks", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-5, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5, help="interval between image sampling")
parser.add_argument("--model_save_interval", type=int, default=200, help="interval between model saves")
parser.add_argument("--load_g", type=str, default=None, help="generator model to load")
parser.add_argument("--load_d", type=str, default=None, help="discriminator model to load")
parser.add_argument("--i_c", type=float, default=0.5, help="")
parser.add_argument("--init_beta", type=float, default=0, help="")
parser.add_argument("--loss_fct", type=str, default="standard", help="standard or vdb")


opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
print("GPU available:", cuda)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
#

class Generator(torch.nn.Module):
    def __init__(self, feature_size):
        super(Generator, self).__init__()

        self.feature_size = feature_size

        # z: 1 -> 2 -> 4 -> 24 -> ... -> 24
        # x: 1 -> 2 -> 6 -> 12 -> 24 -> 48 -> 100 -> 200 -> 400 -> 800 -> 1600

        # y: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 512



        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(opt.latent_dim, (self.feature_size * 16) * 2 * 2, bias=True),
            torch.nn.BatchNorm1d(self.feature_size * 16 * 2 * 2),
            torch.nn.ReLU()
        )  # (1, 2, 2)

        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size * 16, self.feature_size * 8, kernel_size=(3, 4, 4), stride=(2, 2, 2),
                                     padding=(1, 0, 1)),
            torch.nn.BatchNorm3d(self.feature_size * 8),
            torch.nn.ReLU()
        )  # (1, 6, 4)

        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size * 8, self.feature_size * 8, kernel_size=(5, 4, 4), stride=(2, 2, 2),
                                     padding=(2, 1, 1)),
            torch.nn.BatchNorm3d(self.feature_size * 8),
            torch.nn.ReLU()
        )  # (1, 12, 8)
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size * 8, self.feature_size * 4, kernel_size=(5, 4, 4), stride=(2, 2, 2),
                                     padding=(2, 1, 1)),
            torch.nn.BatchNorm3d(self.feature_size * 4),
            torch.nn.ReLU()
        )  # (1, 24, 16)

        # keep z dim constant (at 8)
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size * 4, self.feature_size * 4, kernel_size=(5, 4, 4), stride=(1, 2, 2),
                                     padding=(2, 0, 1)),
            torch.nn.BatchNorm3d(self.feature_size * 4),
            torch.nn.ReLU()
        )  # (1, 50, 32)

        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size * 4, self.feature_size * 2, kernel_size=(5, 4, 4), stride=(1, 2, 2),
                                     padding=(2, 1, 1)),
            torch.nn.BatchNorm3d(self.feature_size * 2),
            torch.nn.ReLU()
        )  # (1, 100, 64)

        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size * 2, self.feature_size * 2, kernel_size=(5, 4, 4), stride=(1, 2, 2),
                                     padding=(2, 1, 1)),
            torch.nn.BatchNorm3d(self.feature_size * 2),
            torch.nn.ReLU()
        )  # (1, 200, 128)

        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size * 2, self.feature_size, kernel_size=(5, 4, 4), stride=(1, 2, 2),
                                     padding=(2, 1, 1)),
            torch.nn.BatchNorm3d(self.feature_size),
            torch.nn.ReLU()
        )  # (1, 400, 256)

        self.layer9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size, self.feature_size, kernel_size=(5, 4, 4), stride=(1, 2, 2),
                                     padding=(2, 1, 1)),
            torch.nn.BatchNorm3d(self.feature_size),
            torch.nn.ReLU()

        )  # (1, 800, 512)

        self.layer10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.feature_size, 1, kernel_size=(5, 4, 5), stride=(1, 2, 1), padding=(2, 1, 2)),
            # torch.nn.BatchNorm3d(self.feature_size),
            torch.nn.Tanh()
        )  # (1, 1600, 512)

    def forward(self, x):
        out = x.view(-1, opt.latent_dim)
        # print("input size:", out.size())

        out = self.fc1(out)
        out = out.view(-1, self.feature_size * 16, 1, 2, 2)
        # print("after layer 1:",out.size())

        out = self.layer2(out)
        # print("after layer 2:",out.size())
        out = self.layer3(out)
        # print("after layer 3:",out.size()

        out = self.layer4(out)

        # print("after layer 4:",out.size())
        out = self.layer5(out)
        # print("after layer 5:",out.size())
        #
        out = self.layer6(out)
        # print("after layer 6:",out.size())
        out = self.layer7(out)
        # print("after layer 7:",out.size())
        out = self.layer8(out)
        # print("after layer 8:",out.size())
        out = self.layer9(out)
        # print("after layer 9:",out.size())
        out = self.layer10(out)
        # print("after layer 10:",out.size())

        return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

    # inverse of generator
    # z: 1 -> 2 -> 4 -> 24 -> ... -> 24
    # x: 1 -> 2 -> 6 -> 12 -> 24 -> 48 -> 100 -> 200 -> 400 -> 800 -> 1600
    # y: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 512


class Discriminator(nn.Module):
    def __init__(self, feature_size):
        super(Discriminator, self).__init__()


        self.feature_size = feature_size


        self.layer1 = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv3d(1, self.feature_size, kernel_size=3, stride=2, padding=(1, 1, 1))),
            torch.nn.MaxPool3d(kernel_size=(1, 2, 2)),
            torch.nn.ReLU()
        )  # (1, 400, 128)

        self.layer2 = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv3d(self.feature_size, self.feature_size * 2, kernel_size=3, stride=2, padding=(1, 1, 1))),
            torch.nn.MaxPool3d(kernel_size=(1, 2, 2)),
            torch.nn.ReLU()
        )  # (1, 100, 64)

        self.layer4 = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv3d(self.feature_size * 2, self.feature_size * 4, kernel_size=3, stride=(1, 2, 1), padding=(1, 1, 1))),
            torch.nn.MaxPool3d(kernel_size=(1, 2, 2)),
            torch.nn.ReLU()
        )  # (1, 25, 16)

        self.layer5 = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv3d(self.feature_size * 4, self.feature_size * 8, kernel_size=3, stride=2, padding=(1, 0, 1))),
            torch.nn.ReLU()
        )  # (1, 12, 8)

        self.flatten = Flatten()

        self.fc1 = torch.nn.Sequential(
            spectral_norm(torch.nn.Linear(self.feature_size * 8 * 12 * 8, 256, bias=True)),
            torch.nn.ReLU()
        )

        self.fc2 = torch.nn.Sequential(
            spectral_norm(torch.nn.Linear(256, 256, bias=True)),
            # torch.nn.ReLU()
        )

        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(128, 1, bias=True),
            torch.nn.Sigmoid()
        )

        self.fc4 = torch.nn.Sequential(
            spectral_norm(torch.nn.Linear(256, 1, bias=True)),
            torch.nn.Sigmoid()
        )

    def forward(self, img, mean_mode=True):
        #print(img)
        # out = x.view(-1, opt.latent_dim, 1, 1, 1)
        out = self.layer1(img)
        # print("after layer 1:",out.size())  # torch.Size([100, 512, 4, 4, 4])
        out = self.layer2(out)
        # print("after layer 2:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        # out = self.layer3(out)
        # print("after layer 3:",out.size())  # torch.Size([100, 256, 8, 8, 8])

        out = self.layer4(out)

        # print("after layer 4:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer5(out)
        # print("after layer 5:",out.size())  # torch.Size([100, 256, 8, 8, 8])

        out = self.flatten(out)

        out = self.fc1(out)
        #print(out)
        # print("after fc1:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.fc2(out)

        if opt.loss_fct == "standard":
            out = self.fc4(out)
            return out, None, None

        # VDB
        parameters = self.flatten(out)
        halfpoint = parameters.shape[-1] // 2
        mus, sigmas = parameters[:, :halfpoint], parameters[:, halfpoint:],
        sigmas = torch.sigmoid(sigmas)


        if mean_mode:
            out = torch.randn_like(mus).to(img.device) * sigmas + mus
        else:
            out = mus



        out = self.fc3(out)

        return out, mus, sigmas

def bottleneck_loss(mus, sigmas, i_c, alpha=1e-8):
    # print(torch.sum(0.5 * (mus ** 2 + sigmas ** 2), dim=1), "----",
    #       mus.shape[1],
    #       torch.sum(torch.log(sigmas ** 2 + alpha), dim=1))
    #
    # a = torch.sum(0.5 * (mus ** 2 + sigmas ** 2), dim=1)
    # b = mus.shape[1]
    # c = torch.sum(torch.log(sigmas ** 2 + alpha), dim=1)
    #print("Sigmas: {}, {}".format(sigmas, sigmas.shape))
    #print("Mus: {}, {}".format(mus, mus.shape))
    kl_divergence = torch.sum(0.5 * (mus ** 2 + sigmas ** 2)
                              - 0.5
                              - torch.log(sigmas ** 2 + alpha), dim=1)

    # calculate the bottleneck loss:
    bl = (torch.mean(kl_divergence) - i_c)

    # return the bottleneck_loss:
    return bl

# def bottleneck_loss(mus, sigmas, i_c, alpha=1e-8):
#
#     kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
#                                   - torch.log((sigmas ** 2) + alpha) - 1, dim=1))
#
#     # calculate the bottleneck loss:
#     bl = (torch.mean(kl_divergence) - i_c)
#
#     # return the bottleneck_loss:
#     return bl

def optimize_beta(beta, bnl):
    #print("BNL")
    #print(bnl)
    beta_new  = max(0, beta + (1e-6 * bnl.item()))
    return beta_new


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
if opt.load_g:
    generator = torch.load(opt.load_g)
else:
    generator = Generator(feature_size=64)

if opt.load_g:
    discriminator = torch.load(opt.load_d)
else:
    discriminator = Discriminator(feature_size=64)


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

dataloader = torch.utils.data.DataLoader(
    MRIDataset(
        csv_file="annotations_slices_best.csv",
        root_dir="wbmri_slices_best",
        # transform=transforms.Compose(
        #     [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        # ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
iter_start_time = time.time()
beta = opt.init_beta


for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):




        iter_time = time.time() - iter_start_time
        iter_start_time = time.time()


        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # print(valid, fake)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor)).unsqueeze(1).unsqueeze(1)

        # print("image shape", imgs.shape)

        # -----------------
        #  Train Generator
        # -----------------


        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)



        d_prediction, mus, sigmas = discriminator(gen_imgs, mean_mode=False)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(d_prediction, valid)

        # print("calculated g_loss")
        optimizer_G.zero_grad()

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------


        # Measure discriminator's ability to classify real from generated samples
        r_d_prediction, r_mus, r_sigmas = discriminator(real_imgs)
        real_loss = adversarial_loss(r_d_prediction, valid)

        f_d_prediction, f_mus, f_sigmas = discriminator(gen_imgs.detach())
        fake_loss = adversarial_loss(f_d_prediction, fake)




        #print(bottle_neck_loss, "=========")

        # d_loss = (real_loss + fake_loss) / 2 + bottle_neck_loss
        if opt.loss_fct == "standard":
            d_loss = (real_loss + fake_loss) / 2
        elif opt.loss_fct == "vdb":
            bottle_neck_loss = bottleneck_loss(
                torch.cat((r_mus, f_mus), dim=0),
                torch.cat((r_sigmas, f_sigmas), dim=0),
                opt.i_c
            )

            d_loss = (real_loss + fake_loss) / 2 + beta * bottle_neck_loss

            beta = optimize_beta(beta, bottle_neck_loss)

        optimizer_D.zero_grad()

        d_loss.backward()
        optimizer_D.step()



        #print(beta)
        if opt.loss_fct == "standard":
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Iter time: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), iter_time)
            )
        elif opt.loss_fct == "vdb":

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Iter time: %f][Bottleneck loss: %f] [Beta: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), iter_time, bottle_neck_loss, beta)
            )

        batches_done = epoch * len(dataloader) + i

        if batches_done % opt.sample_interval == 0:

            im = gen_imgs.cpu().detach().numpy()[0, 0, 0, :, :]
            #imwrite("g_z/epoch_{}_batch_{}.png".format(epoch, i), (im*255).astype(np.uint8))
            matplotlib.image.imsave("g_z/epoch_{}_batch_{}.png".format(epoch, i), im, cmap='gray')
            # plt.imshow(im, cmap="gray")
            # plt.draw()
            # plt.pause(0.001)

        if batches_done % opt.model_save_interval == 0:
            torch.save(generator.state_dict(), "networks/mri_dcgan_generator_epoch_{}".format(epoch))
            torch.save(discriminator.state_dict(), "networks/mri_dcgan_discriminator_epoch_{}".format(epoch))
