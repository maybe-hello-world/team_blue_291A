import numpy as np


def combination(img1: np.ndarray, img2: np.ndarray, prc1: float, prc2: float) -> np.ndarray:
    """
    Combines the two images into one with given percentages
    Args:
        img1: np.ndarray, any int/float 2-3d image
        img2: np.ndarray, any int/float 2-3d image
        prc1: float, any fraction of first image in result
        prc2: float, any fraction of second image in result

    Returns:
        np.ndarray int, [0-255]
    """
    assert img1.shape == img2.shape
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    res = img1 * prc1 + img2 * prc2
    res = (res - res.min()) / (res.max() - res.min()) * 255
    return res
