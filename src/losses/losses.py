import torch
import torch.nn.functional as F
def fourier_space_loss(pred, gt, parts = "both"):
    """
    Computes the Fourier space loss between predicted and ground truth images.

    Args:
    - pred (torch.Tensor): The predicted image tensor.
    - gt (torch.Tensor): The ground truth image tensor.

    Returns:
    - torch.Tensor: The total Fourier space loss.

    This function computes the Fourier space loss between predicted and ground truth images.
    Based on the paper: Fourier Space Losses for Efficient Perceptual Image Super-Resolution
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
    fft_gt = torch.fft.rfft2(gt)
    fft_pred = torch.fft.rfft2(pred)

    # Calculate the absolute value and angle (phase) of the FFT coefficients for ground truth and predicted images
    fft_gt_abs = fft_gt.abs()
    fft_gt_angle = fft_gt.angle()
    fft_pred_abs = fft_pred.abs()
    fft_pred_angle = fft_pred.angle()

    # Compute the absolute loss and angle (phase) loss between predicted and ground truth FFT coefficients
    loss_abs = torch.abs(torch.subtract(fft_pred_abs, fft_gt_abs)).mean()
    loss_angle = torch.abs(torch.subtract(fft_pred_angle, fft_gt_angle)).mean()

    # Combine the absolute loss and angle (phase) loss with equal weights to get the total loss
    if parts == "both":
        total_loss = loss_abs + loss_angle
    elif parts == "abs":
        total_loss = loss_abs
    elif parts == "phase":
        total_loss == loss_angle
    return total_loss