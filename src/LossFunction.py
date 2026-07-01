import torch
from torch import nn

def kl_loss_function(mu, log_sigma):
    kl=0.5*(mu**2 + torch.exp(2*log_sigma)-1-2*log_sigma)
    return torch.sum(kl) # sto sommando (batch_size * 128) valori


def loss_function(reconstructed, original, mu, log_sigma, beta=5):
    
    reconstruction_loss_function=nn.BCELoss(reduction='sum') # somma di (batch_size * 3 * 64 * 64) valori
    return reconstruction_loss_function(reconstructed, original) + \
           beta*kl_loss_function(mu, log_sigma)
