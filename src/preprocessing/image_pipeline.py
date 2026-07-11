import cv2
import numpy as np
from skimage.filters import threshold_niblack


def deskew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line.ravel()
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -25 < angle < 25:
                angles.append(angle)

        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
    return image


def preprocess(roi: np.ndarray, visualize: bool = False, use_otsu: bool = True) -> np.ndarray | tuple:
    steps = {}

    min_width  = 200
    min_height = 60
    roi_h, roi_w = roi.shape[:2]
    if roi_w < min_width or roi_h < min_height:
        scale = max(min_width / roi_w, min_height / roi_h)
        new_w = int(roi_w * scale)
        new_h = int(roi_h * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


    roi = deskew(roi)
    steps['0_deskew'] = roi.copy()


    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    steps['1_gray'] = gray.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    steps['2_clahe'] = enhanced.copy()


    h_roi = enhanced.shape[0]
    w_size = max(11, int(h_roi * 0.5)) 
    if w_size % 2 == 0: 
        w_size += 1   
    thresh = threshold_niblack(enhanced, window_size=w_size, k=0.2)
    binary = (enhanced < thresh).astype(np.uint8) * 255
        
    steps['3_binary'] = binary.copy()

    h_roi = enhanced.shape[0]
    k_size = max(1, int(h_roi * 0.03))

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    steps['4_morphology'] = cleaned.copy()

    if visualize:
        return cleaned, steps
    return cleaned
