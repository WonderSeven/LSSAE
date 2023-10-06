import torch
import torch.nn.functional as F
from .types_ import *


def kl_divergence(latent_space_a, latent_space_b):
    return torch.mean(torch.distributions.kl_divergence(latent_space_a, latent_space_b))


def temporal_smooth_loss(latent_variables: Tensor, batch_first=True):
    if batch_first:
        return F.l1_loss(latent_variables[:, 1:, :], latent_variables[:, :-1, :], reduction='mean')
    else:
        return F.l1_loss(latent_variables[1:, :, :], latent_variables[-1:, :, :], reduction='mean')
