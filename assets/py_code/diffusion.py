import torch
from typing import List, Tuple
import matplotlib.pyplot as plt


def linear_schedule(n_diffusion_steps: int, device: str = "cpu") -> torch.Tensor:
    """Linear noise scheduler as described in [1].

    References
    ----------
    .. [1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel.
        "Denoising diffusion probabilistic models."
        Advances in neural information processing systems 33 (2020): 6840-6851.
    """
    # diffusion scheduler as described in "Denoising diffusion probabilistic models"
    # https://arxiv.org/abs/2006.11239
    diffusion_rate = torch.linspace(1e-4, 1e-2, n_diffusion_steps, device=device)

    return torch.cumprod(1 - diffusion_rate, dim=0)


def ddim_statistics(
    t: int,
    t_prev: int,
    alphas_cumprod: List[float],
    eta: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute DDIM coefficients as described in [1].

    It reads::

        x_t_prev = coef_x_t * x_t + coef_x_0 * x_0 + std * noise

    Return
    ------
    coef_x_t, coef_x_0, std

    References
    ----------
    .. [1] Song, Jiaming, Chenlin Meng, and Stefano Ermon.
        "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).
    """
    acp_t_prev_0 = alphas_cumprod[t_prev] / alphas_cumprod[0]
    acp_t_t_prev = alphas_cumprod[t] / alphas_cumprod[t_prev]
    acp_t_0 = alphas_cumprod[t] / alphas_cumprod[0]

    std = eta * ((1 - acp_t_t_prev) * (1 - acp_t_prev_0) / (1 - acp_t_0)) ** 0.5
    coef_x_t = ((1 - acp_t_prev_0 - std**2) / (1 - acp_t_0)) ** 0.5
    coef_x_0 = acp_t_prev_0**0.5 - coef_x_t * acp_t_0**0.5

    return coef_x_t, coef_x_0, std


def plot_score_vector_field(
    score_func,
    t: int,
    x_range: List[float],
    y_range: List[float],
    device="cpu",
    ax=None,
):
    if ax is None:
        ax = plt
    if isinstance(t, int):
        t = torch.tensor(t, device=device)

    # make mesh that covers the (x_range * y_range)
    n_x_points, n_y_points = len(x_range), len(y_range)
    X, Y = torch.meshgrid(x_range, y_range)

    # compute scores over the mesh
    points = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    points = points.to(device)
    score_values = score_func(points, t)

    # make the result compatible with quiver plot
    U = score_values[:, 0].view(n_x_points, n_y_points)
    V = score_values[:, 1].view(n_x_points, n_y_points)
    U, V = U.cpu(), V.cpu()

    ax.quiver(X, Y, U, V, alpha=0.5, headwidth=3, label="score")
