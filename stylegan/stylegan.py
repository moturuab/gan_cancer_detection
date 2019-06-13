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

from models import *
from models import StyleBased_Generator

os.makedirs("g_z", exist_ok=True)
os.makedirs("networks", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=5e-5, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=512, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=5, help="interval between image sampling")
parser.add_argument("--model_save_interval", type=int, default=200, help="interval between model saves")
parser.add_argument("--load_g", type=str, default=None, help="generator model to load")
parser.add_argument("--load_d", type=str, default=None, help="discriminator model to load")

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
print("GPU available:", cuda)



# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
if opt.load_g:
    generator = torch.load(opt.load_g)
else:
    generator = StyleBased_Generator(n_fc=8, dim_latent=512)

if opt.load_g:
    discriminator = torch.load(opt.load_d)
else:
    discriminator = Discriminator(feature_size=64)


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

dataloader = torch.utils.data.DataLoader(
    MRIDataset(
        csv_file="annotations_slices_opt.csv",
        root_dir="../wbmri_slices_opt",
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


for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        iter_time = time.time() - iter_start_time
        iter_start_time = time.time()


        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)


        # Configure input
        real_imgs = Variable(imgs.type(Tensor)).unsqueeze(1)


        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)


        d_prediction = discriminator(gen_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(d_prediction, valid)

        optimizer_G.zero_grad()

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Measure discriminator's ability to classify real from generated samples
        r_d_prediction = discriminator(real_imgs)
        real_loss = adversarial_loss(r_d_prediction, valid)

        f_d_prediction = discriminator(gen_imgs.detach())
        fake_loss = adversarial_loss(f_d_prediction, fake)


        d_loss = (real_loss + fake_loss) / 2
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Iter time: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), iter_time)
        )

        optimizer_D.zero_grad()

        d_loss.backward()
        optimizer_D.step()


        batches_done = epoch * len(dataloader) + i

        if batches_done % opt.sample_interval == 0:

            im = gen_imgs.cpu().detach().numpy()[0, 0, :, :]
            #imwrite("g_z/epoch_{}_batch_{}.png".format(epoch, i), (im*255).astype(np.uint8))
            matplotlib.image.imsave("g_z/epoch_{}_batch_{}.png".format(epoch, i), im, cmap='gray')
            plt.imshow(im, cmap="gray", vmin=0, vmax=1)
            plt.draw()
            plt.pause(0.001)

        if batches_done % opt.model_save_interval == 0:
            torch.save(generator.state_dict(), "networks/mri_dcgan_generator_epoch_{}".format(epoch))
            torch.save(discriminator.state_dict(), "networks/mri_dcgan_discriminator_epoch_{}".format(epoch))
