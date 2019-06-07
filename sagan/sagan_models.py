import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_
import numpy as np

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)

class Self_Attention(nn.Module):
    def __init__(self, input_dim, activation):
        super(Self_Attention,self).__init__()
        self.channel_in = input_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=input_dim, out_channels=input_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_dim, out_channels=input_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs:
                x : input feature maps (Batch size x # of channels x W x H)
        :param x:
        :return:
        """
        m_batch_size, num_channels, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batch_size,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batch_size,-1,width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batch_size,-1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batch_size, num_channels, width, height)

        out = self.gamma*out + x
        return out, attention

class Generator(nn.Module):

    def __init__(self, batch_size, image_size=(1600, 512), z_dim=100,conv_dim=64):
        super(Generator, self).__init__()
        print(image_size)
        self.im_height = image_size[0]
        self.im_width = image_size[1]
        self.conv_dim = conv_dim
        linear = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer6 = []
        layer7 = []
        last = []

        linear.append(spectral_norm(nn.Linear(in_features=z_dim, out_features=conv_dim*4*1*1)))

        layer1.append(spectral_norm(nn.Conv2d(self.conv_dim*4, self.conv_dim*4, kernel_size=3)))
        layer1.append(nn.BatchNorm2d(conv_dim*4))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * 4

        layer2.append(spectral_norm(nn.Conv2d(curr_dim, int(curr_dim / 2), kernel_size=3)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(spectral_norm(nn.Conv2d(curr_dim, int(curr_dim / 2), kernel_size=3)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer4.append(spectral_norm(nn.Conv2d(curr_dim, int(curr_dim / 2), kernel_size=3)))
        layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer4.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer5.append(spectral_norm(nn.Conv2d(curr_dim, int(curr_dim / 2), kernel_size=3)))
        layer5.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer5.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer6.append(spectral_norm(nn.Conv2d(curr_dim, int(curr_dim), kernel_size=3)))
        layer6.append(nn.BatchNorm2d(int(curr_dim)))
        layer6.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer7.append(spectral_norm(nn.Conv2d(curr_dim, int(curr_dim), kernel_size=3)))
        layer7.append(nn.BatchNorm2d(int(curr_dim)))
        layer7.append(nn.ReLU())

        last.append(spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=3)))
        last.append(nn.Tanh())

        self.sn_linear = nn.Sequential(*linear)
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)
        self.l6 = nn.Sequential(*layer6)
        self.l7 = nn.Sequential(*layer7)
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attention(32, 'relu')
        self.attn2 = Self_Attention(16, 'relu')
        self.attn3 = Self_Attention(8, 'relu')
        self.attn4 = Self_Attention(8, 'relu')

        self.apply(init_weights)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.sn_linear(z) # batch_size x conv_dim x 1 x 1
        out = out.view(-1, self.conv_dim*8,1,1)
        out = F.interpolate(out, scale_factor=(5,4), mode='nearest') # batch_size x cojnv_dim*16 x 5 x 4
        layer1_out = self.l1(out)
        layer1_out_upsample = F.interpolate(layer1_out, scale_factor=4, mode='nearest') # batch_size x conv_dim*16 x 25 x 8
        layer2_out = self.l2(layer1_out_upsample)
        layer2_out_upsample = F.interpolate(layer2_out, scale_factor=4, mode='nearest') # batch_size x conv_dim*8 x 50 x 16
        layer3_out = self.l3(layer2_out_upsample)
        layer3_out_upsample = F.interpolate(layer3_out, scale_factor=4, mode='nearest') # batch_size x conv_dim*4 x 100 x 32
        layer3_attn,p1 = self.attn1(layer3_out_upsample)
        layer4_out = self.l4(layer3_attn)
        layer4_out_upsample = F.interpolate(layer4_out, scale_factor=2,
                                            mode='nearest')  # batch_size x conv_dim*2 x 200 x 64
        layer4_attn, p2 = self.attn2(layer4_out_upsample)
        layer5_out = self.l5(layer4_attn)
        layer5_out_upsample = F.interpolate(layer5_out, scale_factor=2,
                                            mode='nearest')  # batch_size x conv_dim x 400 x 128
        layer5_attn, p3 = self.attn3(layer5_out_upsample)
        layer6_out = self.l6(layer5_attn)
        layer6_out_upsample = F.interpolate(layer6_out, scale_factor=2,
                                            mode='nearest')  # batch_size x conv_dim x 800 x 256
        layer6_attn, p4 = self.attn4(layer6_out_upsample)
        layer7_out = self.l6(layer6_attn)
        layer7_out_upsample = F.interpolate(layer7_out, scale_factor=2,
                                            mode='nearest')  # batch_size x conv_dim x 1600 x 512
        last = self.last(layer7_out_upsample)

        return last, p1, p2, p3, p4

class Discriminator(nn.Module):
    def __init__(self,batch_size, image_size=(1600,512), d_conv_dim=64):
        super(Discriminator, self).__init__()
        self.im_height = image_size[0]
        self.im_width = image_size[1]
        self.conv_dim = d_conv_dim
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer6 = []
        layer7 = []
        last = []

        layer1.append(spectral_norm(nn.Conv2d(1, self.conv_dim, kernel_size=3)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = self.conv_dim

        layer2.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3)))
        layer2.append(nn.MaxPool2d(2))
        layer2.append(nn.LeakyReLU())

        curr_dim = curr_dim * 2

        layer3.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3)))
        layer3.append(nn.MaxPool2d(2))
        layer3.append(nn.LeakyReLU())

        curr_dim = curr_dim * 2


        layer4.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3)))
        layer4.append(nn.MaxPool2d(2))
        layer4.append(nn.LeakyReLU())

        curr_dim = curr_dim * 2

        layer5.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3)))
        layer5.append(nn.MaxPool2d(2))
        layer5.append(nn.LeakyReLU())

        curr_dim = curr_dim * 2

        layer6.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim, kernel_size=3)))
        layer6.append(nn.MaxPool2d(2))
        layer6.append(nn.LeakyReLU())

        curr_dim = curr_dim * 2

        layer7.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim, kernel_size=3)))
        layer7.append(nn.MaxPool2d(5,4))
        layer7.append(nn.LeakyReLU())

        last.append(nn.Conv2d(curr_dim, 1, kernel_size=3))

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)
        self.l6 = nn.Sequential(*layer6)
        self.l7 = nn.Sequential(*layer7)
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attention(32, 'relu')
        self.attn2 = Self_Attention(64, 'relu')
        self.attn3 = Self_Attention(128, 'relu')

        self.apply(init_weights)

    def forward(self,x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.l5(out)
        out, p2 = self.attn3(out)
        out = self.l6(out)
        out = self.l7(out)
        out = self.last(out)

        return out.squeeze(), p1, p2





