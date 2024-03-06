#!/usr/bin/env python
import pytest
import numpy as np
from tabpy.metric import calc_psnr


@pytest.mark.parametrize(
    "img1, img2, expected",
    [
        # Test case 1: Identical images
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            float("inf"),
        ),
        # Test case 2: Different images
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
            -9.5424,
        ),
        # Test case 3: Noisy image is a scaled version of the noise-free image
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[0.5, 1, 1.5], [2, 2.5, 3]]),
            0.2323,
        ),
    ],
)
def test_calc_psnr(img1, img2, expected):
    rst = calc_psnr(img1, img2)
    np.testing.assert_almost_equal(rst, expected, decimal=1)


if __name__ == "__main__":
    pytest.main([__file__])
