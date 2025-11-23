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
from torch.nn.utils import spectral_norm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Input simple rescaling
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Output upsample via deconvolution
        self.upsample = nn.ConvTranspose2d(out_ch,out_ch,kernel_size=4,stride=2,padding=1) #nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        shortcut = self.shortcut(self.upscale(x))
        #x = self.upsample(x)
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x = self.upsample(F.relu(self.bn3(x)))

        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.upsample(x))
        """
        #return F.relu(x + shortcut)
        return x + shortcut


# -----------------------------------------
#       ResNet Generator 256×256
# -----------------------------------------
class GeneratorResNet(nn.Module):
    def __init__(self, z_dim = opt.latent_dim , cond_dim=opt.n_classes, img_channels=3):
        super().__init__()

        #self.init_size = opt.img_size  
        #self.l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.n_classes, 16 * self.init_size ** 2))
        self.fc = nn.Linear(z_dim + cond_dim, 64 * 4 * 4)

        self.res1 = ResBlockUp(64, 64)   #   4 → 8
        self.res2 = ResBlockUp(64, 128)   #   8 → 16
        self.res3 = ResBlockUp(128, 128)   #  16 → 32
        self.res4 = ResBlockUp(128, 64)   #  32 → 64
        self.res5 = ResBlockUp(64, 32)    #  64 → 128
        self.res6 = ResBlockUp(32, 32)     # 128 → 256

        self.final = nn.Sequential(
            nn.Conv2d(32, img_channels, 3, padding=1),
            nn.Sigmoid()   # Use Tanh() if your dataset is [-1,1]
        )

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 64, 4, 4)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        img = self.final(x)
        return img


##############################
#        Discriminator
##############################


class DiscriminatorU(nn.Module):
    def __init__(self, in_channels=3):
        super(DiscriminatorU, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                #layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.BatchNorm2d(out_filters))
            layers += [nn.LeakyReLU(0.2, inplace=True),nn.Dropout2d(0.05)]
            #ayers.append()
            #layers.append(nn.Dropout2d(0.15))
            
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 128),
            *discriminator_block(128, 64),
            nn.ZeroPad2d((1, 0, 1, 0)),
            #nn.Conv2d(512, 1, 4, padding=1, bias=False)
            #nn.Conv2d(128, 64, 4, padding=1),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.15),
            nn.Conv2d(64, 4, 4, padding=1, bias=False)
        )

        self.fc_img = nn.Sequential(
            nn.Linear(8*8*4, opt.n_classes),
            nn.LeakyReLU(0.2)
        )

        #self.fc_cond = nn.Sequential(
        #    nn.Linear(opt.n_classes, 64),
        #    nn.LeakyReLU(0.2)
        #)

        self.adv_layer = nn.Sequential(
            nn.Linear(2*opt.n_classes,64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )

        # Adversarial layer with HxW output
        #self.adv_layer = nn.Sequential(
        #      nn.Linear(128 * ds_size ** 2 + opt.n_classes, 1),
              #nn.Conv2d(2*opt.n_classes, 1, 3, padding=1, bias=True)
        #      nn.Sigmoid()
        #    )
    def forward(self, img, cond):
        out = self.model(img)
        #cond_emb = self.fc_cond(cond)
        out_emb = self.fc_img(out.view(out.shape[0],-1))
        #out = out.view(out.shape[0], -1)
        #d_in = torch.cat((out.view(out.size(0), -1), self.cond_emb(cond)), -1)
        #d_in = torch.cat((out.view(out.size(0), -1), cond), -1)
        #validity = self.adv_layer(d_in)
        #condw = cond.view(cond.shape[0],cond.shape[1],1,1)
        #cond2d = condw*torch.ones(out.shape[0],opt.n_classes,out.shape[2],out.shape[3]).cuda()
        #diff = torch.square(cond - out_emb)
        d_in = torch.cat((out_emb,cond),1)
        #validity = self.adv_layer(d_in)
        validity = self.adv_layer(d_in)
        return validity


#--------------------------------------------------------------------

##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        
        # Init size and gen from latent var
        self.init_size = opt.img_size  # init image size could be set to 64
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.n_classes, 16 * self.init_size ** 2))

        # Upsample the original latent feats (x4)
        self.upsample_layer = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2)
        ) # Output upsampled features (C=32)

        # Unet blocks
        self.down1 = UNetDown(32, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        #self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 256, normalize=False, dropout=0.5)
        #self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        
        self.up1 = UNetUp(256, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(768, 256)                # + d2    → 384 ch  (FIXED)
        self.up6 = UNetUp(384, 128)                # + d1    → 192 ch  (FIXED)

        # Final image generator
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, out_channels, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, cond):
        gen_input = torch.cat((cond, z), -1)
        out = self.l1(gen_input) # flattened input features (C=16)
        x = out.view(out.shape[0],16, self.init_size, self.init_size)# latent input feats
        x = self.upsample_layer(x) # 64 -> 256 (C=32)
        ## (B,C,H,W)
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)


#--------------------------------------------------------------------


# ------------------------------
#   CONDITIONAL BATCHNORM
# ------------------------------
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = spectral_norm(nn.Linear(cond_dim, num_features))
        self.beta  = spectral_norm(nn.Linear(cond_dim, num_features))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma(y).view(y.size(0), -1, 1, 1)
        beta  = self.beta(y).view(y.size(0), -1, 1, 1)
        return out * (1 + gamma) + beta


# ------------------------------
#       GENERATOR BLOCK
# ------------------------------
class GBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, upsample=True):
        super().__init__()
        self.upsample = upsample

        self.cbn1 = ConditionalBatchNorm2d(in_ch, cond_dim)
        self.cbn2 = ConditionalBatchNorm2d(out_ch, cond_dim)

        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))

        self.learnable_sc = (in_ch != out_ch) or upsample
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))

    def shortcut(self, x):
        out = x
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        if self.learnable_sc:
            out = self.conv_sc(out)
        return out

    def forward(self, x, y):
        sc = self.shortcut(x)

        h = self.cbn1(x, y)
        h = F.relu(h)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)

        h = self.cbn2(h, y)
        h = F.relu(h)
        h = self.conv2(h)

        return sc + h


# ------------------------------
#      TINY GENERATOR
# ------------------------------
class TinyBigGANGenerator(nn.Module):
    def __init__(self, z_dim=100, cond_dim=72, img_channels=3):
        super().__init__()

        self.fc = spectral_norm(nn.Linear(z_dim + cond_dim, 64 * 4 * 4))

        # very small channels
        self.block1 = GBlock(64, 64, cond_dim)     # 4 → 8
        self.block2 = GBlock(64, 64, cond_dim)     # 8 → 16
        self.block3 = GBlock(64, 32, cond_dim)     # 16 → 32
        self.block4 = GBlock(32, 32, cond_dim)     # 32 → 64
        self.block5 = GBlock(32, 16, cond_dim)     # 64 → 128
        self.block6 = GBlock(16, 16, cond_dim)     # 128 → 256

        self.bn_final = nn.BatchNorm2d(16)
        self.conv_final = spectral_norm(nn.Conv2d(16, img_channels, 3, padding=1))

        self.out_act = nn.Sigmoid()

    def forward(self, z, cond):
        h = torch.cat([z, cond], dim=1)
        h = self.fc(h)
        h = h.view(h.size(0), 64, 4, 4)

        h = self.block1(h, cond)
        h = self.block2(h, cond)
        h = self.block3(h, cond)
        h = self.block4(h, cond)
        h = self.block5(h, cond)
        h = self.block6(h, cond)

        h = self.bn_final(h)
        h = F.relu(h)
        h = self.conv_final(h)

        return self.out_act(h)   # [0,1] output

#--------------------------------------------------------------------

# ------------------------------
#     DISCRIMINATOR BLOCK
# ------------------------------
class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample

        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))

        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))

    def shortcut(self, x):
        out = x
        if self.learnable_sc:
            out = self.conv_sc(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        return out

    def forward(self, x):
        sc = self.shortcut(x)

        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)

        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return sc + h


# ------------------------------
#  TINY PROJECTION DISCRIMINATOR
# ------------------------------
class TinyBigGANDiscriminator(nn.Module):
    def __init__(self, img_channels=3, cond_dim=72):
        super().__init__()

        # smaller channels
        self.block1 = DBlock(img_channels, 32, downsample=True)   # 256 → 128
        self.block2 = DBlock(32, 64, downsample=True)             # 128 → 64
        self.block3 = DBlock(64, 64, downsample=True)             # 64 → 32
        self.block4 = DBlock(64, 128, downsample=True)            # 32 → 16
        self.block5 = DBlock(128, 128, downsample=True)           # 16 → 8

        self.feat_dim = 128

        self.linear = spectral_norm(nn.Linear(self.feat_dim, 1))
        self.embed_y = spectral_norm(nn.Linear(cond_dim, self.feat_dim))

    def forward(self, x, cond):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)

        h = F.relu(h)
        h = h.sum(dim=[2,3])      # global sum pool → (B,128)

        out_uncond = self.linear(h)
        proj = (h * self.embed_y(cond)).sum(dim=1, keepdim=True)

        return out_uncond + proj   # raw logits
