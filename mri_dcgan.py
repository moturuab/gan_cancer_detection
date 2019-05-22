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


os.makedirs("images", exist_ok=True)

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
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         self.init_size = opt.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # self.args = args

        # padd = (0, 0, 0)
        # if self.cube_len == 32:
            # padd = (1,1,  1)
        self.cube_len = 1

        # z: 1 -> 2 -> 4 -> 8 -> ... -> 8 
        # x: 1 -> 2 -> 6 -> 12 -> 24 -> 45 -> 90 -> 181 -> 362 -> 724 -> 1448
        # y: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 512

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(opt.latent_dim, self.cube_len*8, kernel_size=4, stride=(1, 1, 1), padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.ReLU()
        ) # (2, 2, 2)

        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=(2, 2, 2), padding=(1, 0, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.ReLU()
        ) # (4, 6, 4)

        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=(2, 2, 2), padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.ReLU()
        ) # (8, 12, 8)
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=(5, 4, 4), stride=(1, 2, 2), padding=(2, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        ) # (8, 24, 16)

        # keep z dim constant (at 8)
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, self.cube_len, kernel_size=(5, 5, 4), stride=(1, 2, 2), padding=(2, 3, 1)), 
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        ) # (8, 45, 32)

        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, self.cube_len, kernel_size=(5, 4, 4), stride=(1, 2, 2), padding=(2, 1, 1)),            
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        ) # (8, 90, 64)

        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, self.cube_len, kernel_size=(5, 5, 4), stride=(1, 2, 2), padding=(2, 1, 1)),            
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        ) # (8, 181, 128)

        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, self.cube_len, kernel_size=(5, 4, 4), stride=(1, 2, 2), padding=(2, 1, 1)),            
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        ) # (8, 362, 256)

        self.layer9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, self.cube_len, kernel_size=(5, 4, 4), stride=(1, 2, 2), padding=(2, 1, 1)),            
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()

        ) # (8, 724, 512)

        self.layer10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, self.cube_len, kernel_size=(5, 4, 5), stride=(1, 2, 1), padding=(2, 1, 2)),              
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.Sigmoid()
        ) # (8, 1448, 512)

    def forward(self, x):
        out = x.view(-1, opt.latent_dim, 1, 1, 1)
        print("input size:", out.size())  # torch.Size([100, 200, 1, 1, 1])
        out = self.layer1(out)
        print("after layer 1:",out.size())  # torch.Size([100, 512, 4, 4, 4])
        out = self.layer2(out)
        print("after layer 2:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer3(out)
        print("after layer 3:",out.size())  # torch.Size([100, 256, 8, 8, 8])

        out = self.layer4(out)

        print("after layer 4:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer5(out)
        print("after layer 5:",out.size())  # torch.Size([100, 256, 8, 8, 8])

        out = self.layer6(out)
        print("after layer 6:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer7(out)
        print("after layer 7:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer8(out)
        print("after layer 8:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer9(out)
        print("after layer 9:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer10(out)
        print("after layer 10:",out.size())  # torch.Size([100, 256, 8, 8, 8])

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


# class DenseNet3D(nn.Module):
#     r"""Densenet-BC model class, based on
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
#     Args:
#         growth_rate (int) - how many filters to add each layer (`k` in paper)
#         block_config (list of 4 ints) - how many layers in each pooling block
#         num_init_features (int) - the number of filters to learn in the first convolution layer
#         bn_size (int) - multiplicative factor for number of bottle neck layers
#           (i.e. bn_size * k features in the bottleneck layer)
#         drop_rate (float) - dropout rate after each dense layer
#         num_classes (int) - number of classification classes
#     """
#     def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
#                  num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

#         super(DenseNet3D, self).__init__()

#         # First convolution
#         self.features = nn.Sequential(OrderedDict([
#             ('conv0', nn.Conv3d(1, num_init_features, kernel_size=(3, 7, 7), stride=2, padding=(1, 3, 3), bias=False)),
#             ('norm0', nn.BatchNorm3d(num_init_features)),
#             ('relu0', nn.ReLU(inplace=True)),
#             ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
#         ]))

#         # Each denseblock
#         num_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
#                                 bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
#             self.features.add_module('denseblock%d' % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 num_features = num_features // 2

#         # Final batch norm
#         self.features.add_module('norm5', nn.BatchNorm3d(num_features))

#         # Linear layer
#         self.classifier = nn.Linear(num_features, num_classes)

#     def forward(self, x):
#         features = self.features(x)
#         out = F.relu(features, inplace=True)
#         out = F.avg_pool3d(out, kernel_size=(1,7,7)).view(features.size(0), -1)
#         out = self.classifier(out)
        # return out



# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = nn.Sequential(
#             *discriminator_block(opt.channels, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         # The height and width of downsampled image
#         ds_size = opt.img_size // 2 ** 4
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         validity = self.adv_layer(out)

#         return validity




    # inverse of generator
    # z: 1 -> 2 -> 4 -> 8 -> ... -> 8 
    # x: 1 -> 2 -> 6 -> 12 -> 24 -> 45 -> 90 -> 181 -> 362 -> 724 -> 1448
    # y: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 512
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()


        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=(1, 1, 1)),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            torch.nn.ReLU()
        ) # (2, 362, 128)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=(1, 1, 1)),
            # torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            torch.nn.ReLU()
        ) # (1, 181, 64)

        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=(1, 1, 1)),
        #     torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
        #     torch.nn.ReLU()
        # ) # (1, 181, 64)
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 16, kernel_size=3, stride=(1, 2, 1), padding=(1, 0, 1)),
            torch.nn.MaxPool3d(kernel_size=(1, 2, 2)),
            torch.nn.ReLU()
        ) # (1, 45, 32)
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=(1, 1, 1)),
            torch.nn.ReLU()
        ) # (1, 23, 16)

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(5888, 128, bias=True),
            torch.nn.ReLU()
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.ReLU()
        )

        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(128, 2, bias=True),
            torch.nn.Sigmoid()
        )
        
        # # The height and width of downsampled image
        # ds_size = opt.img_size // 2 ** 4
        # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        # out = x.view(-1, opt.latent_dim, 1, 1, 1)
        out = self.layer1(img)
        print("after layer 1:",out.size())  # torch.Size([100, 512, 4, 4, 4])
        out = self.layer2(out)
        print("after layer 2:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        # out = self.layer3(out)
        # print("after layer 3:",out.size())  # torch.Size([100, 256, 8, 8, 8])

        out = self.layer4(out)

        print("after layer 4:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer5(out)
        print("after layer 5:",out.size())  # torch.Size([100, 256, 8, 8, 8])


        out = self.fc1(out)
        print("after fc1:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.fc2(out)
        print("after fc2:",out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.fc3(out)
        print("after fc3:",out.size())  # torch.Size([100, 256, 8, 8, 8])


        return validity



# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()

generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
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

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths

        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)



        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        
        print("image shape", imgs.shape)



        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)
        print("generated images")

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        print("calculated g_loss")

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)



torch.save(generator.state_dict(), "mri_dcgan_model")
