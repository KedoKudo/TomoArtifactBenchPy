#!/usr/bin/env python
"""Helper functions for evaluate the performance of ring artifacts removal."""
import tomopy
import numpy as np
import matplotlib.pyplot as plt
from ntabpy.synthetic import generate_sinogram, simulate_detector_gain_error
from typing import Tuple


def recon_from_sinogram(
    sinogram: np.ndarray,
    thetas_deg: np.ndarray,
    filter_name: str = "shepp",
) -> np.ndarray:
    """Perform reconstruction with given sinogram and theta.

    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram.
    thetas_deg : np.ndarray
        Projection angles in degrees.

    Returns
    -------
    recon : np.ndarray
        Reconstructed image.

    Example
    -------
    >>> sinogram = np.random.rand(360, 256)
    >>> thetas_deg = np.arange(0, 360, 1)
    >>> recon = recon_from_sinogram(sinogram, thetas_deg)
    >>> print(recon.shape)
    (256, 256)
    """
    # expand sinogram to 3D
    proj = sinogram[:, np.newaxis, :]
    # perform reconstruction
    recon = tomopy.recon(
        proj,
        theta=np.deg2rad(thetas_deg),  # tomopy requires theta in radians
        center=None,  # synthetic data center is always the center of the image
        algorithm="gridrec",  # use gridrec algorithm for its speed
        filter_name=filter_name,
    )

    return recon[0, :, :]


def tomo_round_trip_2d(
    input_img: np.ndarray,
    scan_step: float,
    detector_gain_range: Tuple[float, float] = (0.9, 1.1),
    detector_gain_error: float = 0.1,
    show_plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a round trip tomography simulation on a 2D image.

    Parameters
    ----------
    input_img : np.ndarray
        The input 2D image to perform tomography on.
    scan_step : float
        The step size of the tomography scan, in degrees.
    show_plot : bool, optional
        Whether to show the plot of the sinogram and reconstruction, by default True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The sinogram, reconstruction, gain, reconstruction of gain, and the original sinogram.

    Example
    -------
    >>> img = np.random.rand(256, 256)
    >>> sinogram, recon, gain, recon_gain, sino_org = tomo_round_trip_2d(img, 1)
    >>> print(sinogram.shape, recon.shape, gain.shape, recon_gain.shape, sino_org)
    (360, 256) (256, 256) (360, 256) (256, 256) (360, 256)
    """
    # generate sinogram
    sinogram_org, thetas_deg = generate_sinogram(input_img, scan_step)

    # add detector gain error to sinogram
    sinogram, detector_gain = simulate_detector_gain_error(
        sinogram_org,
        detector_gain_range,
        detector_gain_error,
    )

    # perform reconstruction
    recon_sino = recon_from_sinogram(sinogram, thetas_deg)

    # perform reconstruction of the gain error
    recon_gain = recon_from_sinogram(detector_gain, thetas_deg)

    # show the plot
    if show_plot:
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        axs[0, 0].imshow(input_img, cmap="gray")
        axs[0, 0].set_title("Input Image")
        axs[0, 1].imshow(sinogram_org, cmap="gray")
        axs[0, 1].set_title("Sinogram")

        axs[1, 0].imshow(recon_gain, cmap="gray")
        axs[1, 0].set_title("Reconstruction of Gain Error")
        axs[1, 1].imshow(detector_gain, cmap="gray")
        axs[1, 1].set_title("Gain Error")

        axs[2, 0].imshow(recon_sino, cmap="gray")
        axs[2, 0].set_title("Reconstruction")
        axs[2, 1].imshow(sinogram, cmap="gray")
        axs[2, 1].set_title("Sinogram + Gain Error")

        fig.tight_layout()

        # print the data range
        print("Input Image: ", input_img.min(), input_img.max())
        print("Sinogram: ", sinogram.min(), sinogram.max())
        print("Reconstruction: ", recon_sino.min(), recon_sino.max())
        print("Gain Error: ", detector_gain.min(), detector_gain.max())
        print("Reconstruction of Gain Error: ", recon_gain.min(), recon_gain.max())

    return sinogram, recon_sino, detector_gain, recon_gain, sinogram_org
