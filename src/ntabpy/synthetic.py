#!/usr/bin/env python
"""Generate synthetic data for testing purposes."""
import numpy as np
from skimage.transform import radon
from typing import Tuple


def generate_sinogram(
    input_img: np.ndarray,
    scan_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate sinogram from input image.

    Parameters
    ----------
    input_img : np.ndarray
        Input image.
    scan_step : float
        Scan step in degrees.

    Returns
    -------
    sinogram : np.ndarray
        Generated sinogram.
    theta : np.ndarray
        Projection angles in degrees.

    Example
    -------
    >>> img = np.random.rand(256, 256)
    >>> sinogram, thetas_deg = generate_sinogram(img, 1)
    >>> print(sinogram.shape, thetas_deg.shape)
    (360, 256) (360,)
    """
    # prepare thetas_deg
    thetas_deg = np.arange(-180, 180, scan_step)

    # prepare sinogram
    # perform virtual projection via radon transform
    sinogram = radon(
        input_img,
        theta=thetas_deg,
        circle=False,  # do not clip the image to get the best recon quality.
    ).T  # transpose to get the sinogram in the correct orientation for tomopy

    return sinogram, thetas_deg


def simulate_detector_gain_error(
    sinogram: np.ndarray,
    detector_gain_range: Tuple[float, float],
    detector_gain_error: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate detector gain error.

    Parameters
    ----------
    sinogram : np.ndarray
        Input sinogram.
    detector_gain_range : Tuple[float, float]
        Detector gain range.
    detector_gain_error : float
        Detector gain error, along time axis.

    Returns
    -------
    sinogram : np.ndarray
        Sinogram with detector gain error.
    detector_gain : np.ndarray
        Detector gain.

    Example
    -------
    >>> img = np.random.rand(256, 256)
    >>> sinogram, thetas_deg = generate_sinogram(img, 1)
    >>> sinogram, detector_gain = simulate_detector_gain_error(
    ...     sinogram,
    ...     (0.9, 1.1),
    ...     0.1,
    ... )
    >>> print(sinogram.shape, detector_gain.shape)
    (360, 256) (360, 256)
    """
    # prepare detector_gain
    detector_gain = np.random.uniform(
        detector_gain_range[0],
        detector_gain_range[1],
        sinogram.shape[1],
    )
    detector_gain = np.ones(sinogram.shape) * detector_gain

    # simulate detector gain vary slightly along time axis
    if detector_gain_error != 0.0:
        detector_gain = np.random.normal(
            detector_gain,
            detector_gain * detector_gain_error,
        )

    # apply detector_gain
    sinogram = sinogram * detector_gain

    # rescale sinogram to [0, 1]
    sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min()) + 1e-8

    # convert to float32
    sinogram = sinogram.astype(np.float32)
    detector_gain = detector_gain.astype(np.float32)

    return sinogram, detector_gain


if __name__ == "__main__":
    img = np.random.rand(256, 256)
    sinogram, thetas_deg = generate_sinogram(img, 1)
    print(sinogram.shape, thetas_deg.shape)
    print(sinogram[0, :5])
    print(thetas_deg[:5])
    print(sinogram[-1, :5])
    print(thetas_deg[-5:])
