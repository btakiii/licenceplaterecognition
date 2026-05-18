import numpy as np
from skimage.feature import hog


HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


def extract_hog(char_image: np.ndarray,
                return_image: bool = False) -> np.ndarray | tuple:
    if char_image.shape != (64, 32):
        import cv2
        char_image = cv2.resize(char_image, (32, 64))

    result = hog(
        char_image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=return_image,
        feature_vector=True
    )

    if return_image:
        features, hog_image = result
        return features, hog_image

    return result
