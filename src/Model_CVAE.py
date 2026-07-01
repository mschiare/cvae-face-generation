import torch
import os

from torch import nn
from torch.nn.modules.activation import LeakyReLU

from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

num_classes = 3
latent_dim = 128
class AutoEncoder(nn.Module):
    
    def __init__(self):
        
        super().__init__()
         
          
 
        self.encoder = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(3 + num_classes, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # la media e la varianza vengono calcolate per ogni canale, in questo caso ho 256 canali. per ogni canale considero lo stesso canale di ogni elemento del batch.
            nn.BatchNorm2d(256), # va sempre specificato il numero di canali
            nn.LeakyReLU(0.2),
            
            nn.Flatten() #da [256, 4, 4] a [4096] 
        )
        
        self.fc_mu = nn.Linear(4096, latent_dim) # avrà matrice di pesi W = 128*4096

        # voglio che la rete mi dia il logaritmo della varianza perché applicando in seguito
        # l'esponenziale posso assicurarmi che la varianza sia positiva.
        self.fc_logvar = nn.Linear(4096, latent_dim) # # avrà matrice di pesi W = 128*4096

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim + num_classes, 4096), # W = 4096 * (128+3)
            
            nn.LeakyReLU(0.2),

            nn.Unflatten(1, (256, 4, 4)), # Ricostruisce il cubetto 256x4x4
            
            # Layer 1: 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 32x32 -> 64x64
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Output tra 0 e 1 per le immagini

            )

        
    def forward(self, x, cond):

        # cond[:,:,None,None] passo da [batch_size, 3] -> [batch_size, 3, 1, 1]
        # expand vado da [batch_size, 3, 1, 1] a [batch_size, 3, 64, 64]
        econd=cond[:,:,None,None].expand(-1,-1,x.shape[2],x.shape[3])

        # concateno lungo la dimensione dei canali. 
        # x -> [64, 3, 64, 64] econd -> [64, 3, 64, 64] 
    
        x=torch.cat((x,econd), dim=1) # x_finale -> [64, 6, 64, 64]

        out=self.encoder(x) # [64, 4096] 
        mu=self.fc_mu(out)  # [64,128]
        log_sigma=self.fc_logvar(out) # [64,128]
        eps=torch.randn_like(log_sigma) # [64,128]
        z=mu+eps*torch.exp(log_sigma) # [64, 128]
        z=torch.cat((z,cond),dim=1)   # [64, 131 (128+3)]
        y=self.decoder(z) # [64, 3, 64, 64]
        return y, mu, log_sigma
