import torch
import os

from BalancedBatch import get_balanced_sampler
from Model_CVAE import AutoEncoder
from GenerateFaces import generate_faces
from Training_epoch import training_epoch

from torch.utils.data import DataLoader
from torchvision import transforms as T

from torchvision.datasets import CelebA

device = 'cuda'

output_dir = '/home/M.CHIARELOTTO/progetto/Training_6'
save_dir = '/home/M.CHIARELOTTO/progetto/Training_6/Weights'
os.makedirs(output_dir, exist_ok=True)

transform = T.Compose([
    T.CenterCrop(128),
    T.Resize((64, 64)),
    T.ToTensor(),
])

training_set = CelebA(
    root="/home/pfoggia/GenerativeAI/CELEBA",
    split="all",
    transform=transform,
    download= False
)

# Creazione del DataLoader bilanciato
# NOTA: Quando usi un sampler, shuffle deve essere False!
sampler = get_balanced_sampler(training_set)
train_loader = DataLoader(training_set, batch_size=64, sampler=sampler, num_workers=4, shuffle = False, drop_last=True)


model=AutoEncoder()
state_dict = torch.load("/home/M.CHIARELOTTO/progetto/Training_4/Weights/epoch_53_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
model=model.to(device=device)



optimizer=torch.optim.Adam(model.parameters())

epochs = 100
beta = 5

for epoch in range(epochs):

    training_epoch(model, optimizer, train_loader, beta)
    generate_faces(model, epoch, 8, output_dir)
    filename = f"epoch_{epoch+1}_weights.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)