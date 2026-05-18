import cv2
import numpy as np

ASPECT_RATIO_MIN = 0.6
ASPECT_RATIO_MAX = 6.0

MIN_CHAR_HEIGHT = 15

MIN_HEIGHT_RATIO = 0.35
MAX_HEIGHT_RATIO = 0.95

MIN_AREA = 80

NMS_OVERLAP_THRESH = 0.3

MARGIN_LEFT  = 0.12
MARGIN_RIGHT = 0.98

CHAR_RESIZE = (32, 64)

def resize_with_pad(img: np.ndarray, target_size=(32, 64), bg_color=0) -> np.ndarray:
    target_w, target_h = target_size
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return np.full((target_h, target_w), bg_color, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    result = np.full((target_h, target_w), bg_color, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result

def segment_characters(binary_plate: np.ndarray, visualize: bool = False) -> list | tuple:
    plate_h, plate_w = binary_plate.shape[:2]

    contours, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    x_min = int(plate_w * MARGIN_LEFT)
    x_max = int(plate_w * MARGIN_RIGHT)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        cx = x + w // 2
        if cx < x_min or cx > x_max:
            continue

        aspect_ratio = h / w if w > 0 else 0
        
        if not (ASPECT_RATIO_MIN < aspect_ratio < ASPECT_RATIO_MAX):
            continue

        if h < MIN_CHAR_HEIGHT or (w * h) < MIN_AREA:
            continue

        h_ratio = h / plate_h
        if not (MIN_HEIGHT_RATIO < h_ratio < MAX_HEIGHT_RATIO):
            continue

        raw_boxes.append((x, y, w, h))

    consistent_boxes = []
    if raw_boxes:
        median_h = np.median([b[3] for b in raw_boxes])
        for b in raw_boxes:
            _, _, _, h = b
            if 0.75 * median_h <= h <= 1.25 * median_h:
                consistent_boxes.append(b)

    filtered_boxes = _non_max_suppression(consistent_boxes, NMS_OVERLAP_THRESH)

    candidates = []
    for (x, y, w, h) in filtered_boxes:
        pad = 2 # Kicsit kisebb padding, hogy ne lógjon bele a szomszéd betű
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(plate_w, x + w + pad)
        y2 = min(plate_h, y + h + pad)

        char_roi = binary_plate[y1:y2, x1:x2]
        char_resized = resize_with_pad(char_roi, CHAR_RESIZE, bg_color=0)
        candidates.append((x, char_resized))

    candidates.sort(key=lambda c: c[0])

    if visualize:
        annotated = cv2.cvtColor(binary_plate, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (80, 80, 80), 1)
        for (x, y, w, h) in filtered_boxes:
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return candidates, {'annotated': annotated}

    return candidates

def _non_max_suppression(boxes: list[tuple], overlap_thresh: float) -> list[tuple]:
    if not boxes: return []
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    while boxes_sorted:
        current = boxes_sorted.pop(0)
        kept.append(current)
        cx, cy, cw, ch = current
        remaining = []
        for other in boxes_sorted:
            ox, oy, ow, oh = other
            ix1, iy1 = max(cx, ox), max(cy, oy)
            ix2, iy2 = min(cx + cw, ox + ow), min(cy + ch, oy + oh)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union_area = cw * ch + ow * oh - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou < overlap_thresh:
                remaining.append(other)
        boxes_sorted = remaining
    return kept