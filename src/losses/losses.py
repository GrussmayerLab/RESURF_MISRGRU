"""
losses.py

Loss functions used in RESURF/MISRGRU training.

Notes
-----
- All losses assume spatial dimensions are the last two dims (..., H, W).
  This works for shapes like:
    (H, W)
    (C, H, W)
    (B, C, H, W)
    (B, T, H, W)
  because rfft2 operates on the last two dimensions by default.

- `fourier_space_loss` compares magnitude and/or phase of the FFT coefficients.

Reference
---------
Fourier Space Losses for Efficient Perceptual Image Super-Resolution
#https://openaccess.thecvf.com/content/ICCV2021/papers/Fuoli_Fourier_Space_Losses_for_Efficient_Perceptual_Image_Super-Resolution_ICCV_2021_paper.pdf
"""
from typing import Callable, List

import torch
import torch.nn.functional as F
def fourier_space_loss(pred: torch.Tensor, gt: torch.Tensor, parts: str = "both") -> torch.Tensor:    
    """
    Fourier-space loss between predicted and ground-truth images.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted image tensor with shape (..., H, W).
    gt : torch.Tensor
        Ground-truth image tensor with shape (..., H, W).
    parts : {"both","abs","phase"}
        Which FFT components to use:
        - "abs": magnitude only
        - "phase": phase only
        - "both": magnitude + phase

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """

    # # Create a Hann window
    # hann_window_x = torch.hann_window(gt.shape[-1], periodic=False)
    # hann_window_y = torch.hann_window(gt.shape[-2], periodic=False)
    # hann_window = torch.outer(hann_window_x, hann_window_y)
    # # Apply the Hann window to the images
    # hann_window = hann_window.to(gt.device)  # Move to the same device as the images
    # gt_windowed = gt * hann_window
    # pred_windowed = pred * hann_window

    # Compute the 2D Fast Fourier Transform (FFT) of the ground truth and predicted images
    if pred.shape[-2:] != gt.shape[-2:]:
        raise ValueError(f"pred and gt must match in H,W. Got pred={pred.shape}, gt={gt.shape}")

    # Complex FFT over last 2 dims
    fft_gt = torch.fft.rfft2(gt)
    fft_pred = torch.fft.rfft2(pred)

    # Magnitude and phase
    gt_abs = fft_gt.abs()
    gt_phase = fft_gt.angle()
    pred_abs = fft_pred.abs()
    pred_phase = fft_pred.angle()

    loss_abs = (pred_abs - gt_abs).abs().mean()
    loss_phase = (pred_phase - gt_phase).abs().mean()

    if parts == "both":
        return loss_abs + loss_phase
    elif parts == "abs":
        return loss_abs
    elif parts == "phase":
        return loss_phase
    else:
        raise ValueError("parts must be one of {'both','abs','phase'}")


    # # Compute the absolute loss and angle (phase) loss between predicted and ground truth FFT coefficients
    # loss_abs = torch.abs(torch.subtract(fft_pred_abs, fft_gt_abs)).mean()
    # loss_angle = torch.abs(torch.subtract(fft_pred_angle, fft_gt_angle)).mean()


def l1_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    ) -> torch.Tensor:
    """
    Computes the Fourier space and L1 loss between predicted and ground truth images.

    Args:
    - pred (torch.Tensor): The predicted image tensor.
    - gt (torch.Tensor): The ground truth image tensor.

    Returns:
    - torch.Tensor: The total Fourier space and L1 loss.

    This function computes the Fourier space and L1 loss between predicted and ground truth images.
    Based on the paper: Fourier Space Losses for Efficient Perceptual Image Super-Resolution
    """

    L1 = torch.nn.L1Loss()
    L1_loss = L1(pred, gt)
    return L1_loss

def l2_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    ) -> torch.Tensor:
    L2_loss = F.mse_loss(pred,gt)
    return L2_loss

def _trainable_params(model) -> List[torch.nn.Parameter]:
    """Return trainable parameters (requires_grad=True)."""
    return [p for p in model.parameters() if p.requires_grad]


def grad_l2_norm_isolated(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    """
    Compute ||∇_θ loss_fn(model(inputs), targets)||_2 via a fresh forward.
    Does NOT touch the main training graph.
    """
    params = _trainable_params(model)
    # Need grads for this block:
    with torch.enable_grad():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)  # do NOT .detach()
        grads = torch.autograd.grad(
            loss, params,
            create_graph=False,   # no higher-order graph
            retain_graph=False,   # free this graph immediately
            allow_unused=True
        )
        sqsum = None
        for g in grads:
            if g is not None:
                v = g.pow(2).sum()
                sqsum = v if sqsum is None else (sqsum + v)
        if sqsum is None:
            return 0.0
        return float(torch.sqrt(sqsum + 1e-20).item())