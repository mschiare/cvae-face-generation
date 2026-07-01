import torch
from torch.utils.data import  WeightedRandomSampler
import numpy as np

def get_balanced_sampler(dataset):
    """
    Crea un sampler che bilancia il dataset basandosi sugli attributi.
    Assumiamo che dataset.attr sia un tensore (N, 40) o simile dove
    abbiamo le etichette.
    """
    # Indici degli attributi in CelebA: Smiling(31), Male(20), Young(39)
    # Adatta questi indici se hai già filtrato il dataset e hai solo 3 colonne!
    # Esempio: se nel tuo dataset custom hai solo [Smiling, Male, Young], usa [0, 1, 2]
    
    # Recuperiamo le etichette per tutto il dataset
    # (Adatta questa riga in base a come è fatto il tuo dataset object)
    targets = dataset.attr[:, [31, 20, 39]] # Assumendo dataset standard CelebA
    
    # Convertiamo i 3 attributi binari in un'unica classe "gruppo" (da 0 a 7)
    # Esempio: 000=0, 001=1, ... 111=7
    groups = targets[:, 0] * 4 + targets[:, 1] * 2 + targets[:, 2]
    groups = groups.long()

    # Contiamo quante immagini ci sono per ogni gruppo
    group_counts = torch.bincount(groups)
    
    # Calcoliamo il peso per ogni gruppo (inverso della frequenza)
    group_weights = 1. / group_counts.float()
    
    # Assegniamo a ogni immagine il peso del suo gruppo
    sample_weights = group_weights[groups]
    
    # Creiamo il sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler