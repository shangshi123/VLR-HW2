import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    B = discrim_real.size(dim = 0)
    real_label = torch.ones(B,1).cuda()
    fake_label = torch.zeros(B,1).cuda()
    loss1 = torch.nn.BCEWithLogitsLoss(discrim_real,real_label)
    loss2 = torch.nn.BCEWithLogitsLoss(discrim_fake,fake_label)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss1 + loss2


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    B = discrim_fake.size(dim = 0)
    fake_label = torch.zeros(B,1).cuda()
    loss = torch.nn.BCEWithLogitsLoss(1-discrim_fake,fake_label)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
