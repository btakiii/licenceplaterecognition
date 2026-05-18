import cv2
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from src.recognition.hog_extractor import extract_hog

MODEL_PATH   = Path("data/model/svm_model.pkl")
ENCODER_PATH = Path("data/model/label_encoder.pkl")

class CharacterClassifier:
    def __init__(self):
        self.svm     = None
        self.encoder = LabelEncoder()
        self._load_if_exists()

    def _load_if_exists(self):
        if MODEL_PATH.exists() and ENCODER_PATH.exists():
            self.svm     = joblib.load(MODEL_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
            print(f"[INFO] SVM modell betöltve: {MODEL_PATH}")
        else:
            print("[INFO] Nincs betanított modell. Futtasd a train_classifier.py-t.")

    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.svm,     MODEL_PATH)
        joblib.dump(self.encoder, ENCODER_PATH)
        print(f"[INFO] Modell mentve: {MODEL_PATH}")

    def train(self, images: list[np.ndarray], labels: list[str], evaluate: bool = True):
        print(f"[INFO] HOG feature kinyerés {len(images)} képből...")
        features       = np.array([extract_hog(img) for img in images])
        encoded_labels = self.encoder.fit_transform(labels)

        print(f"[INFO] SVM finomhangolás (GridSearch) folyamatban...")
        print(f"       Ez eltarthat egy ideig (több tucat modellt tanít be a háttérben)...")
        
        param_grid = {
            'C': [1, 10, 50, 100],
            'gamma': ['scale', 0.01, 0.001],
            'kernel': ['rbf', 'linear']
        }
        
        grid = GridSearchCV(SVC(decision_function_shape="ovr"), param_grid, cv=3, n_jobs=-1, verbose=1)
        grid.fit(features, encoded_labels)
        
        self.svm = grid.best_estimator_
        print(f"[SIKER] Legjobb paraméterek: {grid.best_params_}")

        if evaluate:
            preds = self.svm.predict(features)
            print("\n--- Tanítási halmazon mért teljesítmény ---")
            print(classification_report(
                encoded_labels, preds,
                target_names=self.encoder.classes_
            ))

    @staticmethod
    def _normalize_polarity(img: np.ndarray) -> np.ndarray:
        if img.mean() < 127:
            return cv2.bitwise_not(img)
        return img

    def predict(self, char_image: np.ndarray) -> str | None:
        if self.svm is None:
            print("[HIBA] Nincs betanított modell!")
            return None

        char_image = self._normalize_polarity(char_image)
        features   = extract_hog(char_image).reshape(1, -1)
        encoded    = self.svm.predict(features)[0]
        return self.encoder.inverse_transform([encoded])[0]

    def predict_batch(self, char_images: list[np.ndarray]) -> list[str]:
        if self.svm is None:
            print("[HIBA] Nincs betanított modell!")
            return []

        normalized = [self._normalize_polarity(img) for img in char_images]
        features   = np.array([extract_hog(img) for img in normalized])
        encoded    = self.svm.predict(features)
        return list(self.encoder.inverse_transform(encoded))