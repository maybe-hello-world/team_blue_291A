import numpy as np
from skimage.filters import threshold_otsu

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

def combination_trial1(sal: np.ndarray, seg:np.ndarray, thre: float) ->np.ndarray:
  """
  Combines two the saliency map and the segmentation map
  Args:
    sal: np.ndarray, int/float 2-3d image
    seg: np.ndarray, int/float 2-3d image
    thre: float, threshold for the saliency map in combination
  returns:
    np.ndarray float 2-3d image
  """
  threshold = np.max(sal)*thre
  sal_retained = sal
  sal_retained[sal<=threshold] = 0
  sal_retained[sal>0] = 255
  sal_norm = cv2.normalize(sal_retained, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  seg_norm = cv2.normalize(seg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  seg_sal = np.where(seg_norm, seg_norm, sal_norm)
  return seg_sal

def combination_trial2(img1, img2, prc1, prc2):
  """
  Combines two images into oe with given percentage where the one the first image will be thresholded to binary
  Args:
    sal: np.ndarray, int/float 2-3d image
    seg: np.ndarray, int/float 2-3d image
    thre: float, threshold for the saliency map in combination
  returns:
    np.ndarray float 2-3d image
  """
  assert img1.shape == img2.shape
  img1 = cv2.threshold(img1,np.mean(img1),255,cv2.THRESH_BINARY)[1]
  img1 = (img1 - img1.min()) / (img1.max() - img1.min())
  img2 = (img2 - img2.min()) / (img2.max() - img2.min())
  res = img1 * prc1 +  img2 * prc2
  res = (res - res.min()) / (res.max() - res.min()) * 255
  return res

def combination_trial3(img1, img2, prc1, prc2):
  """
  Combines two images into one with given percentage where the one the first image will be adding contrast
  Args:
    sal: np.ndarray, int/float 2-3d image
    seg: np.ndarray, int/float 2-3d image
    thre: float, threshold for the saliency map in combination
  returns:
    np.ndarray float 2-3d image
  """
  assert img1.shape == img2.shape
  img1 = cv2.convertScaleAbs(img1, 1.5, 1)
  img1 = (img1 - img1.min()) / (img1.max() - img1.min())
  img2 = (img2 - img2.min()) / (img2.max() - img2.min())
  res = img1 * prc1 +  img2 * prc2
  res = (res - res.min()) / (res.max() - res.min()) * 255
  return res
