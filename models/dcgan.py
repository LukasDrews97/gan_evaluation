from torch import Tensor
import torch.nn as nn
from typing import Tuple

image_sizes = frozenset([28, 32, 64, 128, 256])

class Generator(nn.Module):
    """
    Creates a generator network based on transposed convolutional layers, layer normalization and ReLU activation functions.
    Args:
        noise_dim:
            Dimension of noise as integer number
        img_dim:
            Image size with format (channels, width, height)
    References:
        [1] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
        Alec Radford, Luke Metz, Soumith Chintala
        https://arxiv.org/abs/1511.06434
        [2] Layer Normalization
        Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
        https://arxiv.org/abs/1607.06450
    Raises:
        AssertionError:
            If image size is not one of [28x28, 32x32, 64x64, 128x182, 256x256]
    """
    def __init__(self, noise_dim: int, img_dim: Tuple[int, int, int]) -> None:
        super().__init__()

        assert img_dim[1] in image_sizes
        assert img_dim[1] == img_dim[2]

        self.img_dim = img_dim

        modules = []

        # input shape: [batch_size, 100, 1, 1], output shape: [batch_size, 1024, 4, 4]
        modules.append(self._block(noise_dim, 1024, size=4, kernel_size=4, stride=1, padding=0))
        # input shape: [batch_size, 1024, 4, 4], output shape: [batch_size, 512, 8, 8]
        modules.append(self._block(1024, 512, size=8))
        # input shape: [batch_size, 512, 8, 8], output shape: [batch_size, 256, 16, 16]
        modules.append(self._block(512, 256, size=16))
        # input shape: [batch_size, 256, 8, 8], output shape: [batch_size, 128, 16, 16]
        #modules.append(self._block(256, 128, size=16))

        if img_dim[1] == 28:
            # input shape: [batch_size, 256, 16, 16], output shape: [batch_size, img_dim[0], 28, 28]
            modules.append(nn.ConvTranspose2d(256, img_dim[0], kernel_size=4, stride=2, padding=3, bias=False))

        elif img_dim[1] == 32:
            # input shape: [batch_size, 128, 16, 16], output shape: [batch_size, img_dim[0], 32, 32]
            modules.append(nn.ConvTranspose2d(256, img_dim[0], kernel_size=4, stride=2, padding=1, bias=False))
        elif img_dim[1] in [64, 128, 256]:
            # input shape: [batch_size, 256, 16, 16], output shape: [batch_size, 128, 32, 32]
            modules.append(self._block(256, 128, size=32))

        if img_dim[1] == 64:
            # input shape: [batch_size, 128, 32, 32], output shape: [batch_size, img_dim[0], 64, 64]
            modules.append(nn.ConvTranspose2d(128, img_dim[0], kernel_size=4, stride=2, padding=1, bias=False))
        elif img_dim[1] in [128, 256]:
            # input shape: [batch_size, 128, 32, 32], output shape: [batch_size, 64, 64, 64]
            modules.append(self._block(128, 64, size=64))

        if img_dim[1] == 128:
            # input shape: [batch_size, 64, 64, 64], output shape: [batch_size, img_dim[0], 128, 128]
            modules.append(nn.ConvTranspose2d(64, img_dim[0], kernel_size=4, stride=2, padding=1, bias=False))
        elif img_dim[1] == 256:
            # input shape: [batch_size, 64, 64, 64], output shape: [batch_size, 32, 128, 128]
            modules.append(self._block(64, 32, size=128))
            # input shape: [batch_size, 32, 128, 128], output shape: [batch_size, img_dim[0], 256, 256]
            modules.append(nn.ConvTranspose2d(32, img_dim[0], kernel_size=4, stride=2, padding=1, bias=False))
        
        modules.append(nn.Tanh())


        self.model = nn.Sequential(*modules)

    # kernel_size=4, stride=2, padding=1 doubles the size
    def _block(self, in_channels: int, out_channels: int, size: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.LayerNorm([out_channels, size, size]),                           
            nn.ReLU(True)
        )
        


    def forward(self, noise: Tensor) -> Tensor:
        """Performs forward propagation through the network.
        Args:
            noise: noise tensor for image generation.
        Returns:
            Tensor with generated images.
        """
        return self.model(noise)



class Discriminator(nn.Module):
    """
    Creates a discriminator network based on convolutional layers, layer normalization and LeakyReLU activation functions.
    Expects inputs to be quadratic.
    Possible input sizes are 28x28, 32x32, 64x64, 128x182, 256x256
    Args:
        img_dim:
            Image size with format (channels, width, height)
        batch_size:
            Batch size as integer
        use_sigmoid:
            Whether to use a sigmoid function as final layer
    References:
        [1] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
        Alec Radford, Luke Metz, Soumith Chintala
        https://arxiv.org/abs/1511.06434
        [2] Layer Normalization
        Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
        https://arxiv.org/abs/1607.06450
    Raises:
        AssertionError:
            If image size is not one of [28x28, 32x32, 64x64, 128x182, 256x256]
    """
    def __init__(self, img_dim: Tuple[int, int, int], batch_size: int, use_sigmoid: bool = True) -> None:
        super().__init__()

        assert img_dim[1] in image_sizes
        assert img_dim[1] == img_dim[2]

        self.batch_size = batch_size
        self.img_dim = img_dim
        self.printed = False

        modules = []
        
        if img_dim[1] == 28:
            # input shape: [batch_size, img_dim[0], 28, 28], output shape: [batch_size, 256, 16, 16]
            modules.append(nn.Conv2d(img_dim[0], 256, kernel_size=4, stride=2, padding=3, bias=False))
        
        if img_dim[1] == 32:
            # input shape: [batch_size, img_dim[0], 32, 32], output shape: [batch_size, 256, 16, 16]
            modules.append(nn.Conv2d(img_dim[0], 256, kernel_size=4, stride=2, padding=1, bias=False))

        if img_dim[1] == 64:
            # input shape: [batch_size, img_dim[0], 64, 64], output shape: [batch_size, 128, 32, 32]
            modules.append(nn.Conv2d(img_dim[0], 128, kernel_size=4, stride=2, padding=1, bias=False))

        if img_dim[1] == 128:
            # input shape: [batch_size, img_dim[0], 128, 128], output shape: [batch_size, 64, 64, 64]
            modules.append(nn.Conv2d(img_dim[0], 64, kernel_size=4, stride=2, padding=1, bias=False))
        
        if img_dim[1] == 256:
            # input shape: [batch_size, img_dim[0], 256, 256], output shape: [batch_size, 128, 128, 128]
            modules.append(nn.Conv2d(img_dim[0], 128, kernel_size=4, stride=2, padding=1, bias=False))

        modules.append(nn.LeakyReLU(0.2))

        if img_dim[1] == 256:
            # input shape: [batch_size, 32, 128, 128], output shape: [batch_size, 64, 64, 64]
            modules.append(self._block(32, 64, size=64))

        if img_dim[1] in [128, 256]:
            # input shape: [batch_size, 64, 64, 64], output shape: [batch_size, 128, 32, 32]
            modules.append(self._block(64, 128, size=32))

        if img_dim[1] in [64, 128, 256]:
            # input shape: [batch_size, 128, 32, 32], output shape: [batch_size, 256, 16, 16]
            modules.append(self._block(128, 256, size=16))


        # input shape: [batch_size, 256, 16, 16], output shape: [batch_size, 512, 8, 8]
        modules.append(self._block(256, 512, size=8))
        # input shape: [batch_size, 512, 8, 8], output shape: [batch_size, 1024, 4, 4]
        modules.append(self._block(512, 1024, size=4))
        # input shape: [batch_size, 512, 4, 4], output shape: [batch_size, 1024, 2, 2]
        #modules.append(self._block(512, 1024, size=2))
        # input shape: [batch_size, 1024, 4, 4], output shape: [batch_size, 1, 1, 1]
        modules.append(nn.Conv2d(1024, 1, kernel_size=3, stride=2, padding=0, bias=False))
        
        if use_sigmoid:
            modules.append(nn.Sigmoid())

        self.model = nn.Sequential(*modules)


    # kernel_size=4, stride=2, padding=1 halves the size
    def _block(self, in_channels: int, out_channels: int, size: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.LayerNorm([out_channels, size, size]),
            nn.LeakyReLU(0.2, True)
        )


    def forward(self, imgs: Tensor) -> Tensor:
        """Performs forward propagation through the network.
        Args:
            imgs: 
                Tensor with images to propagate through the network
        Returns:
            Tensor with probabilities.
        """
        return self.model(imgs)


def weights_init(model: nn.Module) -> None:
    """Initialize network weights using normal distributed numbers and bias to 0.
    Args:
        model:
            The model to initialize weights for
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)