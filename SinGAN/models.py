import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1 and classname.find('PreLayerNorm') == -1 and classname.find('PreNorm') == -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y

        
class FullSelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super().__init__()

        # Construct the conv layers
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B * N * C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B * C * N
        energy = torch.bmm(proj_query, proj_key)  # batch matrix-matrix product

        attention = self.softmax(energy)  # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B * C * N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # batch matrix-matrix product
        out = out.view(m_batchsize, C, width, height)  # B * C * W * H

        # Add attention weights onto input
        out = self.gamma * out + x
        return out, attention
        

class WDiscriminatorFSA(nn.Module):
    def __init__(self, opt):
        super(WDiscriminatorFSA, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = FullSelfAttn( max(N, opt.min_nfc))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)
        

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            x,_ = self.attn(x)
        x = self.tail(x)
        return x
        
    

class GeneratorConcatSkip2CleanAddFSA(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAddFSA, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = FullSelfAttn(max(N, opt.min_nfc))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            x,_ = self.attn(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        

from axial_attention import AxialAttention


class AxialWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(AxialWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x) + x
        x = self.tail(x)
        return x


class AxialGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(AxialGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x) + x
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        

class AxialWDiscriminator2(nn.Module):
    def __init__(self, opt):
        super(AxialWDiscriminator2, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x) + x
        x = self.body(x)
        x = self.tail(x)
        return x


class AxialGeneratorConcatSkip2CleanAdd2(nn.Module):
    def __init__(self, opt):
        super(AxialGeneratorConcatSkip2CleanAdd2, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x) + x
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        

        

class AxialWDiscriminator3(nn.Module):
    def __init__(self, opt):
        super(AxialWDiscriminator3, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn1 = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma1 = nn.Parameter(torch.zeros(1))
            self.attn2 = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma2 = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.gamma1 * self.attn1(x) + x
        x = self.body1(x)
        if hasattr(self, 'attn'):
            x = self.gamma2 * self.attn2(x) + x
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        return x


class AxialGeneratorConcatSkip2CleanAdd3(nn.Module):
    def __init__(self, opt):
        super(AxialGeneratorConcatSkip2CleanAdd3, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn1 = self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma1 = nn.Parameter(torch.zeros(1))
            self.attn2 = self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma2 = nn.Parameter(torch.zeros(1))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.gamma1 * self.attn1(x) + x
        x = self.body1(x)
        if hasattr(self, 'attn'):
            x = self.gamma2 * self.attn2(x) + x
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        


class AxialWDiscriminator4(nn.Module):
    def __init__(self, opt):
        super(AxialWDiscriminator4, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn1 = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma1 = nn.Parameter(torch.zeros(1))
            self.attn2 = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma2 = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body1(x)
        x = self.body2(x)
        if hasattr(self, 'attn'):
            x = self.gamma2 * self.attn2(x) + x
        x = self.body3(x)
        if hasattr(self, 'attn'):
            x = self.gamma1 * self.attn1(x) + x
        x = self.tail(x)
        return x


class AxialGeneratorConcatSkip2CleanAdd4(nn.Module):
    def __init__(self, opt):
        super(AxialGeneratorConcatSkip2CleanAdd4, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn1 = self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma1 = nn.Parameter(torch.zeros(1))
            self.attn2 = self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma2 = nn.Parameter(torch.zeros(1))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body1(x)
        x = self.body2(x)
        if hasattr(self, 'attn'):
            x = self.gamma2 * self.attn2(x) + x
        x = self.body3(x)
        if hasattr(self, 'attn'):
            x = self.gamma1 * self.attn1(x) + x
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        

class AxialWDiscriminator5(nn.Module):
    def __init__(self, opt):
        super(AxialWDiscriminator5, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x) + x
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        return x


class AxialGeneratorConcatSkip2CleanAdd5(nn.Module):
    def __init__(self, opt):
        super(AxialGeneratorConcatSkip2CleanAdd5, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn = AxialAttention(
                dim=3,  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=3,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
            
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        if hasattr(self, 'attn'):
           x = self.gamma * self.attn(x) + x
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        


class AxialWDiscriminator6(nn.Module):
    def __init__(self, opt):
        super(AxialWDiscriminator6, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn1 = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma1 = nn.Parameter(torch.zeros(1))
            self.attn2 = AxialAttention(
                dim=3,  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=3,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma2 = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        if hasattr(self, 'attn'):
            x = self.gamma2 * self.attn2(x) + x
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.gamma1 * self.attn1(x) + x
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        return x

        
class AxialGeneratorConcatSkip2CleanAdd6(nn.Module):
    def __init__(self, opt):
        super(AxialGeneratorConcatSkip2CleanAdd6, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn1 = AxialAttention(
                dim=3,  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=3,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma1 = nn.Parameter(torch.zeros(1))
            self.attn2 = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma2 = nn.Parameter(torch.zeros(1))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        if hasattr(self, 'attn'):
           x = self.gamma2 * self.attn2(x) + x
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        if hasattr(self, 'attn'):
           x = self.gamma1 * self.attn1(x) + x
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        
        


class AxialWDiscriminator7(nn.Module):
    def __init__(self, opt):
        super(AxialWDiscriminator7, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn = DecoderAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x.permute([0,2,3,1]).contiguous()).permute([0,3,1,2]).contiguous() + x
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        return x


class AxialGeneratorConcatSkip2CleanAdd7(nn.Module):
    def __init__(self, opt):
        super(AxialGeneratorConcatSkip2CleanAdd7, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        i = 0
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body1 = block
        i = 1
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body2 = block
        i=2
        N = int(opt.nfc / pow(2, (i + 1)))
        block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body3 = block
        if opt.attn == True:
            self.attn = DecoderAxionalLayer(
                dim=3,  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=3,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
            
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        if hasattr(self, 'attn'):
           x = self.gamma * self.attn(x.permute([0,2,3,1]).contiguous()).permute([0,3,1,2]).contiguous() + x
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
                
        

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
      
    return permutations
        

class MySelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        
        return out

# axial attention class
from axial_attention.axial_attention import PermuteToFrom

class MyAxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, MySelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out
        
        
class MyAxialWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(MyAxialWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = MyAxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x) + x
        x = self.tail(x)
        return x


class MyAxialGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(MyAxialGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = self.attn = MyAxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x) + x
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        

class DecoderAxionalLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, dim, dim_index, heads, num_dimensions, sum_axial_out):
        super().__init__()
        self.attn = AxialAttention(dim=dim, dim_index=dim_index,
                                   heads=heads , num_dimensions=num_dimensions,
                                   sum_axial_out=sum_axial_out)
        self.layernorm_attn = nn.LayerNorm([dim], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([dim], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 2*dim, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2*dim, dim, bias=True))

    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
        y = self.attn(X)
        X = self.layernorm_attn(y + X)
        y = self.ffn(X)
        X = self.layernorm_ffn(y + X)
        return X

class AxialDecLWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(AxialDecLWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = DecoderAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.attn(x.permute([0,2,3,1]).contiguous()).permute([0,3,1,2]).contiguous()
        x = self.tail(x)
        return x


class AxialDecLGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(AxialDecLGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = DecoderAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.attn(x.permute([0,2,3,1]).contiguous()).permute([0,3,1,2]).contiguous()
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        

        
class AxialDecLWDiscriminator2(nn.Module):
    def __init__(self, opt):
        super(AxialDecLWDiscriminator2, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = DecoderAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.attn(x.permute([0,2,3,1]).contiguous()).permute([0,3,1,2]).contiguous()
        x = self.body(x)
        x = self.tail(x)
        return x


class AxialDecLGeneratorConcatSkip2CleanAdd2(nn.Module):
    def __init__(self, opt):
        super(AxialDecLGeneratorConcatSkip2CleanAdd2, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = DecoderAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.attn(x.permute([0,2,3,1]).contiguous()).permute([0,3,1,2]).contiguous()
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        


class DecoderFFAxionalLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, dim, dim_index, heads, num_dimensions, sum_axial_out):
        super().__init__()
    
        self.layernorm_ffn = nn.LayerNorm([dim], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 2*dim, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2*dim, dim, bias=True))

    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
      
        y = self.ffn(X)
        X = self.layernorm_ffn(y + X)
        return X
        

class AxialDecLWDiscriminator3(nn.Module):
    def __init__(self, opt):
        super(AxialDecLWDiscriminator3, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = DecoderFFAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.attn(x.permute([0,2,3,1]).contiguous()).permute([0,3,1,2]).contiguous()
        x = self.tail(x)
        return x


class AxialDecLGeneratorConcatSkip2CleanAdd3(nn.Module):
    def __init__(self, opt):
        super(AxialDecLGeneratorConcatSkip2CleanAdd3, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = DecoderFFAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.attn(x.permute([0,2,3,1]).contiguous()).permute([0,3,1,2]).contiguous()
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        


class ImageAttn(nn.Module):
    def __init__(self, in_dim, num_heads, block_length, attn_type='global'): #block length = number of columnes
        super().__init__()
        self.hidden_size = in_dim
        self.kd = in_dim // 2
        self.vd = in_dim
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.block_length = block_length
        self.q_dense = nn.Linear(self.hidden_size, self.kd, bias=False)
        self.k_dense = nn.Linear(self.hidden_size, self.kd, bias=False)
        self.v_dense = nn.Linear(self.hidden_size, self.vd, bias=False)
        self.output_dense = nn.Linear(self.vd, self.hidden_size, bias=False)
        assert self.kd % self.num_heads == 0
        assert self.vd % self.num_heads == 0

    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        return weights @ v

    def forward(self, X):
        X = X.permute([0, 2, 3, 1]).contiguous()
        orig_shape = X.shape
        #X = X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])  # Flatten channels into width
        X = X.view(X.shape[0], -1, X.shape[3])
        q = self.q_dense(X)
        k = self.k_dense(X)
        v = self.v_dense(X)
        # Split to shape [batch_size, num_heads, len, depth / num_heads]
        q = q.view(q.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.num_heads, self.vd // self.num_heads)).permute([0, 2, 1, 3])
        q *= (self.kd // self.num_heads) ** (-0.5)

        if self.attn_type == "global":
            bias = -1e9 * torch.triu(torch.ones(X.shape[1], X.shape[1]), 1).to(X.device)
            result = self.dot_product_attention(q, k, v, bias=bias)
        elif self.attn_type == "local_1d":
            len = X.shape[1]
            blen = self.block_length
            pad = (0, 0, 0, (-len) % self.block_length) # Append to multiple of block length
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)

            bias = -1e9 * torch.triu(torch.ones(blen, blen), 1).to(X.device)
            first_output = self.dot_product_attention(
                q[:,:,:blen,:], k[:,:,:blen,:], v[:,:,:blen,:], bias=bias)

            if q.shape[2] > blen:
                q = q.view(q.shape[0], q.shape[1], -1, blen, q.shape[3])
                k = k.view(k.shape[0], k.shape[1], -1, blen, k.shape[3])
                v = v.view(v.shape[0], v.shape[1], -1, blen, v.shape[3])
                local_k = torch.cat([k[:,:,:-1], k[:,:,1:]], 3) # [batch, nheads, (nblocks - 1), blen * 2, depth]
                local_v = torch.cat([v[:,:,:-1], v[:,:,1:]], 3)
                tail_q = q[:,:,1:]
                bias = -1e9 * torch.triu(torch.ones(blen, 2 * blen), blen + 1).to(X.device)
                tail_output = self.dot_product_attention(tail_q, local_k, local_v, bias=bias)
                tail_output = tail_output.view(tail_output.shape[0], tail_output.shape[1], -1, tail_output.shape[4])
                result = torch.cat([first_output, tail_output], 2)
                result = result[:,:,:X.shape[1],:]
            else:
                result = first_output[:,:,:X.shape[1],:]

        result = result.permute([0, 2, 1, 3]).contiguous()
        result = result.view(result.shape[0:2] + (-1,))
        result = self.output_dense(result)
        result = result.view(orig_shape[0],orig_shape[1], orig_shape[2] ,orig_shape[3])#.permute([0, 3, 1, 2])
        return result
        
        

class ImgAttnWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(ImgAttnWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = ImageAttn(in_dim=max(N, opt.min_nfc), num_heads=4, block_length=max(N, opt.min_nfc))
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x).permute([0, 3, 1, 2]).contiguous() + x
        x = self.tail(x)
        return x


class ImgAttnGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(ImgAttnGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = ImageAttn(in_dim=max(N, opt.min_nfc), num_heads=4, block_length=max(N, opt.min_nfc))
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x).permute([0, 3, 1, 2]).contiguous() + x
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        
        
class DecoderAttnLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, in_dim, num_heads, block_length, dropout=0.1):
        super().__init__()
        self.attn = ImageAttn(in_dim, num_heads, block_length)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm_attn = nn.LayerNorm([in_dim], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([in_dim], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(in_dim, 2*in_dim, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2*in_dim, in_dim, bias=True))


    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
        y = self.attn(X)
        X = X.permute([0, 2, 3, 1])
        X = self.layernorm_attn(self.dropout(y) + X)
        y = self.ffn(X)
        X = self.layernorm_ffn(self.dropout(y) + X)
        return X.permute([0, 3, 1, 2]).contiguous()

#exact image transformer implementation
class ImgTransfromerWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(ImgTransfromerWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)
        if opt.attn == True:
            self.attn = DecoderAttnLayer(
                in_dim = max(N, opt.min_nfc),
                num_heads = 4,
                block_length = max(N, opt.min_nfc),
            )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            x = self.attn(x)
        x = self.tail(x)
        return x


class ImgTransfromerGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(ImgTransfromerGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )
        if opt.attn == True:
            self.attn = DecoderAttnLayer(
                in_dim=max(N, opt.min_nfc),
                num_heads=4,
                block_length=max(N, opt.min_nfc),
            )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            x = self.attn(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        

'''        
from performer_pytorch import SelfAttention, Performer


class PerformerSAWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PerformerSAWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = SelfAttention(
                dim=max(N, opt.min_nfc),
                heads=4,
                causal=False
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x_hat = x.permute([0,2,3,1]).contiguous()
            x_hat = x_hat.view(x_hat.shape[0], -1, x_hat.shape[3])
            x_hat = self.attn(x_hat)
            x_hat = x_hat.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1])
            x_hat = x_hat.permute([0,3,1,2]).contiguous()
            x = self.gamma * x_hat + x
        x = self.tail(x)
        return x


class PerformerSAGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(PerformerSAGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = SelfAttention(
                dim=max(N, opt.min_nfc),
                heads=4,
                causal=False
            )
            self.gamma = nn.Parameter(torch.zeros(1))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x_hat = x.permute([0, 2, 3, 1]).contiguous()
            x_hat = x_hat.view(x_hat.shape[0], -1, x_hat.shape[3])
            x_hat = self.attn(x_hat)
            x_hat = x_hat.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1])
            x_hat = x_hat.permute([0, 3, 1, 2]).contiguous()
            x = self.gamma * x_hat + x
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        
        
        
class PerformerWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PerformerWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = Performer(
                    dim = max(N, opt.min_nfc),
                    dim_head = max(N, opt.min_nfc) // 4,
                    heads = 4,
                    causal = False,
                    depth=1
                    )
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x_hat = x.permute([0, 2, 3, 1]).contiguous()
            x_hat = x_hat.view(x_hat.shape[0], -1, x_hat.shape[3])
            x_hat = self.attn(x_hat)
            x_hat = x_hat.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1])
            x = x_hat.permute([0, 3, 1, 2]).contiguous()
        x = self.tail(x)
        return x


class PerformerGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(PerformerGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = Performer(
                    dim = max(N, opt.min_nfc),
                    dim_head = max(N, opt.min_nfc) // 4,
                    heads = 4,
                    causal = False,
                    depth=1
                    )
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x_hat = x.permute([0, 2, 3, 1]).contiguous()
            x_hat = x_hat.view(x_hat.shape[0], -1, x_hat.shape[3])
            x_hat = self.attn(x_hat)
            x_hat = x_hat.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1])
            x = x_hat.permute([0, 3, 1, 2]).contiguous()
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
'''

        
        
        

        
        
        
        

from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size=[4,4], embed_dim = 48, depth=1, heads=4, mlp_dim =96, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., output_dim=3):
        super().__init__()

        self.btm_pad = - image_size[0] % patch_size[0]
        self.rht_pad = - image_size[1] % patch_size[1]
        self.pad_input = nn.ZeroPad2d((0, self.rht_pad, 0, self.btm_pad))

        img_size = image_size[0] + self.btm_pad, image_size[1] + self.rht_pad

        patch_dim = channels*patch_size[0]*patch_size[1]
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),
            #nn.Linear(patch_dim, embed_dim),
        )

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.dropout = nn.Dropout(emb_dropout)
        dim_head = embed_dim // heads
        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        h = img_size[0] // patch_size[0]
        w = img_size[1] // patch_size[1]
        patch_dim = (patch_dim // channels) * output_dim
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=patch_size[0], p2=patch_size[1], c=output_dim)
        )


    def forward(self, img):
        shape = img.shape
        img = self.pad_input(img)
        x = self.to_patch_embedding(img)
        x = x.transpose(0,1)
        b, n, _ = x.shape

        #x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.transpose(0,1)

        return self.from_patch_embedding(x)[:,:,:shape[2], :shape[3]]

class ViTWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(ViTWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            h, w = opt.cur_real_h_w[0], opt.cur_real_h_w[1]
            self.attn = ViT(image_size=[h, w], output_dim=1)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 3, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        if hasattr(self, 'attn'):
            x = self.attn(x)
        return x


class ViTGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(ViTGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            h, w = opt.cur_real_h_w[0], opt.cur_real_h_w[1]
            self.attn = ViT(image_size=[h,w])
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )
        

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        if hasattr(self, 'attn'):
            x = self.attn(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        
        
class SegAxialDecLWDiscriminator(nn.Module):
    def __init__(self, opt, patch_size=[6, 6]):
        super(SegAxialDecLWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.patch_size = patch_size
        N = int(opt.nfc)
        self.h, self.w = opt.real_image_size
        self.h, self.w = self.h - 8, self.w - 8
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = DecoderAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=32,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)


    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        shape = x.shape
        if hasattr(self, 'attn'):
            h, w = self.h, self.w
            btm_pad = - h % self.patch_size[0]
            rht_pad = - w % self.patch_size[1]
            pad_input = nn.ZeroPad2d((0, rht_pad, 0, btm_pad))

            x = pad_input(x)

            _, c, fh, fw = x.shape
            to_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                   p1=self.patch_size[0], p2=self.patch_size[1], c=c)

            h = fh // self.patch_size[0]
            w = fw // self.patch_size[1]

            x = to_patches(x)
            x = x.transpose(0, 1).view(-1, c, self.patch_size[0], self.patch_size[1])
            x = self.attn(x.permute([0, 2, 3, 1]).contiguous()).permute([0, 3, 1, 2]).contiguous()
            x = x.view(1, x.shape[0], -1)
            from_patches = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                     h=h, w=w, p1=self.patch_size[0], p2=self.patch_size[1], c=c)
            x = from_patches(x)
        x = x[:shape[0], :shape[1], :shape[2], :shape[3]]
        x = self.tail(x)
        return x


class SegAxialDecLGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt, patch_size=[6, 6]):
        super(SegAxialDecLGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.patch_size = patch_size
        N = opt.nfc

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = DecoderAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=3,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=32,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )
        self.h, self.w = opt.fake_image_size
        self.h, self.w = self.h - 8, self.w - 8


    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        shape = x.shape
        if hasattr(self, 'attn'):
            btm_pad = - self.h % self.patch_size[0]
            rht_pad = - self.w % self.patch_size[1]
            pad_input = nn.ZeroPad2d((0, rht_pad, 0, btm_pad))

            x = pad_input(x)

            _, c, fh, fw = x.shape
            to_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                   p1=self.patch_size[0], p2=self.patch_size[1], c=c)

            h = fh // self.patch_size[0]
            w = fw // self.patch_size[1]

            x = to_patches(x)
            x = x.transpose(0, 1).view(-1, c, self.patch_size[0], self.patch_size[1])
            x = self.attn(x.permute([0, 2, 3, 1]).contiguous()).permute([0, 3, 1, 2]).contiguous()
            x = x.view(1, x.shape[0], -1)
            from_patches = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                     h=h, w=w, p1=self.patch_size[0], p2=self.patch_size[1], c=c)
            x = from_patches(x)
        x = x[:shape[0], :shape[1], :shape[2], :shape[3]]
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        
        

        

class ImageAttnGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(ImageAttnGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )
        if opt.attn == True:
            self.attn = ImageAttn(
                in_dim=max(N, opt.min_nfc),
                num_heads=4,
                block_length=max(N, opt.min_nfc),
            )
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.gamma * x + self.attn(x).permute([0, 3, 1, 2])
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        

from linear_attention_transformer.images import ImageLinearAttention



class LinearAttnWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(LinearAttnWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)
        if opt.attn == True:
            self.attn =ImageLinearAttention(
                        chan = max(N, opt.min_nfc), heads = 4,
                        key_dim = 64       # can be decreased to 32 for more memory savings
                        )
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x) + x
        x = self.body(x)
        x = self.tail(x)
        return x


class LinearAttnGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(LinearAttnGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )
        if opt.attn == True:
            self.attn = ImageLinearAttention(
                chan=max(N, opt.min_nfc), heads=4,
                key_dim=64  # can be decreased to 32 for more memory savings
            )
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        x = self.head(x)
        if hasattr(self, 'attn'):
            x = self.gamma * self.attn(x) + x
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y

        
class DecoderLinearAttnLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""

    def __init__(self, chan, heads, key_dim=64):
        super().__init__()
        self.attn = ImageLinearAttention(
            chan=chan, heads=heads,
            key_dim=key_dim  # can be decreased to 32 for more memory savings
        )
        self.layernorm_attn = nn.LayerNorm([chan], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([chan], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(chan, 2 * chan, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2 * chan, chan, bias=True))

    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
        y = self.attn(X)
        X = self.layernorm_attn((y + X).permute([0,2,3,1]).contiguous())
        y = self.ffn(X)
        X = self.layernorm_attn(y + X).permute([0,3,1,2]).contiguous()
        return X


class LinearDecWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(LinearDecWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)
        if opt.attn == True:
            self.attn = DecoderLinearAttnLayer(
                chan=max(N, opt.min_nfc), heads=4,
                key_dim=64  # can be decreased to 32 for more memory savings
            )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.attn(x)
        x = self.tail(x)
        return x


class LinearDecGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(LinearDecGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )
        if opt.attn == True:
            self.attn = DecoderLinearAttnLayer(
                chan=max(N, opt.min_nfc), heads=4,
                key_dim=64  # can be decreased to 32 for more memory savings
            )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.attn(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        
        
        

class SegAxialDecLWDiscriminator(nn.Module):
    def __init__(self, opt, patch_size=[6, 6]):
        super(SegAxialDecLWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.patch_size = patch_size
        N = int(opt.nfc)
        self.h, self.w = opt.real_image_size
        self.h, self.w = self.h - 8, self.w - 8
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = []
            btm_pad = - self.h % self.patch_size[0]
            rht_pad = - self.w % self.patch_size[1]
            self.patches = (self.h + btm_pad) * (self.w + rht_pad) // (patch_size[0] * patch_size[1])
            for i in range(self.patches):
                self.attn.append(
                    DecoderAxionalLayer(
                    dim=max(N, opt.min_nfc),  # embedding dimension
                    dim_index=3,  # where is the embedding dimension
                    # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                    heads=1,  # number of heads for multi-head attention
                    num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                    sum_axial_out=True).to(torch.device('cuda'))
                )
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)


    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        shape = x.shape
        if hasattr(self, 'attn'):
            h, w = self.h, self.w
            btm_pad = - h % self.patch_size[0]
            rht_pad = - w % self.patch_size[1]
            pad_input = nn.ZeroPad2d((0, rht_pad, 0, btm_pad))

            x = pad_input(x)

            _, c, fh, fw = x.shape
            to_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                   p1=self.patch_size[0], p2=self.patch_size[1], c=c)

            h = fh // self.patch_size[0]
            w = fw // self.patch_size[1]

            x = to_patches(x)
            x = x.transpose(0, 1).view(-1, c, self.patch_size[0], self.patch_size[1])
            y = torch.zeros_like(x, device=x.device)
            for i in range(self.patches):
                inp = x[i, :, :, :].unsqueeze(0)
                y[i, :, :, :] = self.attn[i](inp.permute([0, 2, 3, 1]).contiguous()).permute([0, 3, 1, 2]).contiguous()
            y = y.view(1, x.shape[0], -1)
            from_patches = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                     h=h, w=w, p1=self.patch_size[0], p2=self.patch_size[1], c=c)
            y = from_patches(y)
        x = y[:shape[0], :shape[1], :shape[2], :shape[3]]
        x = self.tail(x)
        return x


class SegAxialDecLGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt, patch_size=[6, 6]):
        super(SegAxialDecLGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.patch_size = patch_size
        N = opt.nfc
        self.h, self.w = opt.fake_image_size
        self.h, self.w = self.h - 8, self.w - 8

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = []
            btm_pad = - self.h % self.patch_size[0]
            rht_pad = - self.w % self.patch_size[1]
            self.patches = (self.h + btm_pad) * (self.w + rht_pad) // (patch_size[0] * patch_size[1])
            for i in range(self.patches):
                self.attn.append(
                    DecoderAxionalLayer(
                        dim=max(N, opt.min_nfc),  # embedding dimension
                        dim_index=3,  # where is the embedding dimension
                        # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                        heads=1,  # number of heads for multi-head attention
                        num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                        sum_axial_out=True).to(torch.device('cuda'))
                )
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )



    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        shape = x.shape
        if hasattr(self, 'attn'):
            btm_pad = - self.h % self.patch_size[0]
            rht_pad = - self.w % self.patch_size[1]
            pad_input = nn.ZeroPad2d((0, rht_pad, 0, btm_pad))

            x = pad_input(x)

            _, c, fh, fw = x.shape
            to_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                   p1=self.patch_size[0], p2=self.patch_size[1], c=c)

            h = fh // self.patch_size[0]
            w = fw // self.patch_size[1]

            x = to_patches(x)
            x = x.transpose(0, 1).view(-1, c, self.patch_size[0], self.patch_size[1])
            y_hat = torch.zeros_like(x, device=x.device)
            for i in range(self.patches):
                inp = x[i, :, :, :].unsqueeze(0)
                y_hat[i, :, :, :] = self.attn[i](inp.permute([0, 2, 3, 1]).contiguous()).permute([0, 3, 1, 2]).contiguous()
            y_hat = y_hat.view(1, x.shape[0], -1)

            from_patches = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                     h=h, w=w, p1=self.patch_size[0], p2=self.patch_size[1], c=c)
            y_hat = from_patches(y_hat)
        x = y_hat[:shape[0], :shape[1], :shape[2], :shape[3]]
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y