import torch
from tqdm import tqdm
from networks import Discriminator, Generator
from utils import get_fid, interpolate_latent_space, save_plot

gen = torch.jit.load("data_ls_gan/generator.pt")

interpolate_latent_space(gen, "data_ls_gan/" + "interpolations_fixed.png")