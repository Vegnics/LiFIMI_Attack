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
from matplotlib import pyplot as plt

from gan import GeneratorResNet,GeneratorUNet,TinyBigGANGenerator
from gan import DiscriminatorU,TinyBigGANDiscriminator

os.makedirs("images", exist_ok=True)

# Simple function to parse arguments in Colab
def parse_colab(parse_list):
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
  parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
  parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
  parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
  parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
  parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
  parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
  parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
  parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
  parser.add_argument("--channels", type=int, default=1, help="number of image channels")
  parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
  opt = parser.parse_args(parse_list)
  return opt


arg_list = [
    "--n_epochs", "200",
    "--batch_size", "32",
    "--lr", "0.0002",
    "--b1", "0.5",
    "--b2", "0.999",
    "--n_cpu", "8",
    "--latent_dim", "100",
    "--img_size", "64",
    "--channels", "3",
    "--sample_interval", "400",
    "--n_classes","72"
]

opt = parse_colab(arg_list)
print(opt)

cuda = True if torch.cuda.is_available() else False

# Loss function
#adversarial_loss = torch.nn.MSELoss()
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator
generator = ResNetGenerator() #GeneratorUNet()
discriminator = DiscriminatorU()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)

"""
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
"""

dataset = FFDDatasetCSV("./data_root/warped",csv_file="./data_root/metadata.csv")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs,labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        labels = Variable(labels.type(Tensor))/30.0

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        #gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        #gen_labels = Variable(Tensor(np.random.uniform(low=-30.0,high=30.0,size=(batch_size,opt.n_classes))))

        #gen_labels = Variable(Tensor(np.random.normal(size=(batch_size,opt.n_classes),loc=-2.0,scale=10.0)))

        # sample cond from real labels
        idx = torch.randint(0, labels.size(0), (batch_size,))
        gen_labels = labels[idx].detach()

        # Generate a batch of images
        gen_imgs = generator(z,gen_labels)
        #gen_imgs = generator(z,labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs,labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_labels), fake)
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