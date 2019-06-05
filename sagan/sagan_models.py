import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_
import numpy as np

class Self_Attention(nn.Module):
    def __init__(self, input_dim, activation):
        super(Self_Attention,self).__init__()
        self.channel_in = input_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=input_dim, out_channels=input_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_dim, out_channels=input_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.gamma = nn.parameter(torch.zeros(1))

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
        self.im_height = image_size[0]
        self.im_width = image_size[1]
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

