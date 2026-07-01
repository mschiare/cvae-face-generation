from torchvision.utils import save_image

import torch
import os




def generate_faces(model,epoch, n_cols, output_dir):
    device = 'cuda'
    model.eval()

    specs = [
        [0,  1,  1], # Donna, Sorride, Giovane
        [0, 0, 0], # Donna, Non Sorride, Anziana
        [ 1,  1,  1], # Uomo, Sorride, Giovane
        [ 1, 0, 0], # Uomo, Non Sorride, Anziano
        [ 0, 0, 1], # Donna, Non Sorride, Giovane
        [ 0,  1, 0], # Donna, Sorride, Anziana
        [ 1,  1,  0], # Uomo, Sorride, Anziano
        [ 1,  0,  1], # Uomo, Non Sorride, Giovane


    ]

    z_list = []
    cond_list = []

    with torch.no_grad():
     
     for spec in specs:
        # Estraiamo gli attributi
        is_male, is_smiling, is_young = spec

        # 1. Creiamo il vettore latente (Z)
        # Assicurati che LATENT_DIM sia coerente con il tuo modello (es. 32)
        z = torch.randn(n_cols, 128).to(device)

        # 2. Creiamo i tensori per le classi usando i parametri in input

        m = torch.full((n_cols, 1), float(is_male)).to(device)
        s = torch.full((n_cols, 1), float(is_smiling)).to(device)
        y = torch.full((n_cols, 1), float(is_young)).to(device)

        # 3. Prepariamo la condizione
        cond_raw = torch.cat([m, s, y], dim=1)
  

        z_cond = torch.cat((z, cond_raw), dim=1)
        # Passiamo al decoder (il decoder si aspetta batch, LATENT_DIM + 3)
        generated = model.decoder(z_cond).cpu()


        cond_list.append(generated)

    final_tensor = torch.cat(cond_list, dim=0)



    filename = f"epoch_{epoch+1}_gen.png"
    save_path = os.path.join(output_dir, filename)
    save_image(final_tensor, save_path, nrow=n_cols, padding=2, normalize=False)
   
    print(f"Immagine salvata in: {save_path}")
