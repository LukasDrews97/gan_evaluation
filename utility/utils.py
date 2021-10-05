import torch
from torch import Tensor
from typing import Dict, Tuple
import os

def parse_none_true_false(dict_: Dict) -> Dict:
    """Convert all strings in a dictionary equal to 'None', 'False', 'True' to proper python objects.
    Args:
        dict_:
            Dictionary
    Returns:
        Dictionary with proper python objects
    """
    parse_none = lambda dict: {k: None if v == 'None' else v for k, v in dict.items()}
    parse_true = lambda dict: {k: True if v == 'True' else v for k, v in dict.items()}
    parse_false = lambda dict: {k: False if v == 'False' else v for k, v in dict.items()}
    transformations = [parse_none, parse_true, parse_false]
    for fun in transformations:
        dict_ = fun(dict_)

    return dict_

def normalize_to_0_1(images: Tensor) -> Tensor:
    """Normalize images to interval [0,1].
    Args:
        images:
            Tensor with images
    Returns:
        Tensor with images normalized to [0, 1] interval
    """
    return torch.clamp(((images + 1) / 2.0), 0, 1)

def normalize_to_0_255(images: Tensor) -> Tensor:
    """Normalize images to interval [0,255].
    Args:
        images:
            Tensor with images
    Returns:
        Tensor with images normalized to [0, 255] interval
    """
    return normalize_to_0_1(images) * 255

def adjust_dimensions(imgs: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert two-dimensional greyscale images to three-dimensional RGB images.
    Args:
        imgs:
            Tensor with images
    Returns:
        Image tensor with adjusted dimensions
    """
    # Add empty dimension to greyscale images
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(1)
    # Convert greyscale images to rgb
    if imgs.shape[1] == 1:
        imgs = imgs.expand(-1, 3, -1, -1)

    return imgs

def save_images_to_disk(imgs: Tensor, path: str, file_name: str) -> None:
    """Save images to disk
    Args:
        imgs:
            Tensor containing images
        path:
            Path to save images at
        file_name:
            File name for saved file
    """
    experiment_id = f"experiment{os.environ.get('DET_EXPERIMENT_ID', '_unknown'):0>5}"
    if path[-1] != '/':
        path = path + '/'
    dir_path = path + experiment_id

    os.makedirs(dir_path, exist_ok=True)
    torch.save(imgs, f'{dir_path}/{file_name}')
