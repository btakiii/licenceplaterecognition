

import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[FIGYELMEZTETÉS] Az ultralytics könyvtár nem elérhető. "
          "Fallback kontúr-alapú detektálásra váltás.")


class PlateDetector:
    def __init__(self, model_path: str = None, confidence: float = 0.4):
        self.confidence = confidence

        if YOLO_AVAILABLE:
            custom = Path("data/model/best.pt")
            if model_path:
                self.model = YOLO(model_path)
            elif custom.exists():
                self.model = YOLO(str(custom))
                print(f"[INFO] Egyedi modell betöltve: {custom}")
            else:
                self.model = YOLO("yolov8n.pt")
                print("[INFO] Alap yolov8n.pt modell betöltve.")
        else:
            self.model = None

    def detect(self, image: np.ndarray) -> list[dict]:
        if self.model is not None:
            return self._detect_yolo(image)
        else:
            return self._detect_fallback(image)

    def _detect_yolo(self, image: np.ndarray) -> list[dict]:
        results = self.model(image, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'roi': roi
                })

        return detections

    def _detect_fallback(self, image: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        h_img, w_img = image.shape[:2]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0

            if 2.0 < aspect_ratio < 6.0:
                area = w * h
                if 0.01 * w_img * h_img < area < 0.3 * w_img * h_img:
                    roi = image[y:y+h, x:x+w]
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'confidence': 0.5,
                        'roi': roi
                    })

        detections.sort(key=lambda d: (d['bbox'][2] - d['bbox'][0]) *
                                       (d['bbox'][3] - d['bbox'][1]), reverse=True)
        return detections[:3]
