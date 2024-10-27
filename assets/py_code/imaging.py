import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(img_path, device="cpu") -> torch.Tensor:
    """Load and image as Tensor.

    img_path : str or Path
        the path of the image.
    """
    image = Image.open(img_path)

    im = torch.tensor(np.array(image)).type(torch.FloatTensor).to(device)
    x_origin = ((im - 127.5) / 127.5).squeeze(0)
    x_origin = x_origin.permute(2, 0, 1)

    return x_origin


def display_img_tensor(x: torch.Tensor, ax=None):
    """Plot an image provided as tensor.

    Plot the image in an matplotlib.pyplot axis.

    x : Tensor
        Tensor of shape (3, H, W) whoses coordinate range is [-1, 1].

    ax : pyplot axis, default=None
        If ``None``, the pyplot is used directly.
    """
    if ax is None:
        ax = plt

    # map values [-1, 1] ---> [0, 1]
    x = (x + 1) / 2

    # preprocessing x to be compatible with matplotlib
    x = x.squeeze(0)
    x = x.permute(1, 2, 0)
    x = x.cpu()

    ax.imshow(x)
