#!/usr/bin/env python
"""Metrics to evaluate the performance of ring artifacts removal."""
import numpy as np


def calc_psnr(
    img_noise_free: np.ndarray,
    img_noisy: np.ndarray,
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters
    ----------
    img_noise_free : np.ndarray
        Noise-free image.
    img_noisy : np.ndarray
        Noisy image.

    Returns
    -------
    float
        PSNR value.
    """
    mse = np.mean((img_noise_free / 2 - img_noisy / 2) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


if __name__ == "__main__":
    img_noise_free = np.random.rand(256, 256)
    img_noisy = img_noise_free + np.random.rand(256, 256) * 0.1
    print(calc_psnr(img_noise_free, img_noisy))
