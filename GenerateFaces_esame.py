from GenerateFaces import generate_faces
from Model_CVAE import AutoEncoder
import torch

device = 'cuda'

model=AutoEncoder()
state_dict = torch.load("/home/M.CHIARELOTTO/progetto/Training_4/Weights/epoch_42_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
model=model.to(device=device)

output_dir = '/home/M.CHIARELOTTO/progetto/Generazione'

generate_faces(model, 0, 8, output_dir)