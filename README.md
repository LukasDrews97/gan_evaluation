# Synthetical Image Generation using Generative Adversarial Networks (GANs)
In this project, I generated synthetical images using Non Saturating GANs (NSGANs), Wasserstein-GANs (WGANs) and Deep-Convolutional GANs (DCGANs).
Each model was evaluated using the following metrics:
- Inception Score
- Fréchet Inception Distance
- Kernel Inception Distance
- Unbiased Inception Score & Unbiased Fréchet Inception Distance (https://arxiv.org/pdf/1911.07023v3.pdf)

Using the following datasets:
- MNIST
- Fashion-MNISTs
- Cifar-10
- CelebA

## Project Structure
|File/Folder               |Description|
|---|---|
|`models`|Folder containing the GAN models|
|`train_dcgan.py`|Entry point for training a DCGAN|
|`train_nsgan.py`|Entry point for training a NSGAN|
|`train_wgan.py`|Entry point for training a WGAN|
|`config`|Folder containing config files for determined.ai|
|`metrics`|Folder containing implemented metrics|
|`datasets.py`|Contains all datasets for training|

## Insights

<img src="./imgs/output_27_0.png"/>
