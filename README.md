# Face Generation using Conditional Variational Autoencoders (CVAE)

This repository contains a **PyTorch implementation of a Conditional Variational Autoencoder (CVAE)** trained on the **CelebA** dataset for controlled face generation. The model synthesizes human faces while allowing independent control over three specific attributes: gender (*Male*), smile (*Smiling*), and age (*Young*).

## 🚀 Key Features

- **Conditional CVAE Architecture**: The target attributes condition both the Encoder and the Decoder, but they are processed differently: they are reshaped and spatially expanded to match the image dimensions before entering the Encoder, while they are concatenated directly to the latent vector before entering the Decoder.
- **Dataset Balancing**: Features a custom `WeightedRandomSampler` to actively mitigate the severe native class imbalance (combinations of the 3 target attributes) within CelebA.
- **Beta-VAE Loss**: A tailored loss function combining reconstruction error (`BCELoss`) and Kullback-Leibler Divergence (`KL Divergence`), weighted by a hyperparameter $\beta$. This setup ensures that the model minimizes reconstruction error while simultaneously forcing the latent space to approximate a standard normal distribution, enabling smooth and high-quality face generation from random noise.

---

## 📁 Repository Structure

```text
├── src/
│   ├── Model_CVAE.py           # Model definition (Encoder, Decoder, Reparameterization)
│   ├── LossFunction.py         # Reconstruction and KL Loss calculations
│   ├── BalancedBatch.py        # Dataset balancing logic based on class combinations
│   ├── Training_epoch.py       # Single-epoch training loop logic
│   ├── Train_CVAE.py           # Main script for model training and hyperparameter setup
│   ├── GenerateFaces.py        # Core inference and random latent sampling functions
│   └── GenerateFaces_esame.py  # Simplified script for rapid generation during exam review
├── weigths/
│   ├── weights.pth             # Weigths

└── README.md                   # Project documentation
```

## 🧠 Technical Overview

### Model Architecture
- **Encoder**: Built with 4 convolutional layers (`Conv2d`) with alternating `BatchNorm2d` and `LeakyReLU` activations. Before entering the Encoder, the 3 binary condition attributes are reshaped and spatially expanded to match the image dimensions ($64 \times 64$) and concatenated channel-wise with the input image. The network then outputs the mean ($\mu$) and log-variance ($\log\sigma$) vectors used for the *reparameterization trick*.
- **Latent Space**: The latent dimension is set to `128`.
- **Decoder**: Receives a sampled latent vector $z$ directly concatenated with the raw condition attributes. Through 4 transposed convolutional layers (`ConvTranspose2d`), it reconstructs the original $64 \times 64 \times 3$ image size, ending with a `Sigmoid` activation.

### Loss Function & Optimization
- **Beta-VAE Loss**: A tailored loss function combining reconstruction error (`BCELoss`) and Kullback-Leibler Divergence (`KL Divergence`), weighted by a hyperparameter $\beta$. This setup ensures that the model minimizes the reconstruction error while simultaneously forcing the latent space to approximate a standard normal distribution, enabling smooth and high-quality face generation from random noise.
