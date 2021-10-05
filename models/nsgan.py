import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Tuple


class Generator(nn.Module):
    """
    Creates a generator network based on linear layers, batch normalization and LeakyReLU activation functions.
    Args:
        noise_dim:
            Dimension of noise as integer number
        img_dim:
            Image size with format (channels, width, height)
        batch_size:
            Batch size as integer
    References:
        [1] Generative Adversarial Networks
        Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
        https://arxiv.org/abs/1406.2661
        [2] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        Sergey Ioffe, Christian Szegedy
        https://arxiv.org/abs/1502.03167
    """
    def __init__(self, noise_dim: int, img_dim: Tuple[int, int, int], batch_size: int) -> None:
        super().__init__()
        self.img_dim = img_dim
        self.batch_size = batch_size

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, int(np.prod(img_dim))),
            nn.Tanh()
        )

    def forward(self, noise: Tensor, shape=None) -> Tensor:
        """Performs forward propagation through the network.
            Args:
                noise: 
                    Noise tensor for image generation.
                shape:
                    Output shape for returned images. If None, 
                    output shape will be (batch_size, *img_dim)
            Returns:
                Tensor with generated images.
        """
        if shape is None:
            shape = (self.batch_size, *self.img_dim)
        return self.model(noise).contiguous().view(shape)


class Discriminator(nn.Module):
    """
    Creates a discriminator network based on linear layers, batch normalization and LeakyReLU activation functions.
    Args:
        img_dim:
            Image size with format (channels, width, height)
        batch_size:
            Batch size as integer
    References:
        [1] Generative Adversarial Networks
        Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
        https://arxiv.org/abs/1406.2661
        [2] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        Sergey Ioffe, Christian Szegedy
        https://arxiv.org/abs/1502.03167
    """
    def __init__(self, img_dim: Tuple[int, int, int], batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_dim)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, imgs: Tensor) -> Tensor:
        """Performs forward propagation through the network.
            Args:
                imgs: 
                    Tensor with images to propagate through the network
            Returns:
                Tensor with probabilities
        """
        return self.model(imgs.contiguous().view((self.batch_size, -1)))
