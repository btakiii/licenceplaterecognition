"""
train_classifier.py
-------------------
KNN/SVM osztályozó betanítása szintetikus karakterképekkel.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from src.recognition.knn_classifier import CharacterClassifier
from src.segmentation.char_segmenter import resize_with_pad

CHARACTERS = list("ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")

IMG_WIDTH         = 32
IMG_HEIGHT        = 64
FONT_SIZES        = [28, 32, 36, 40, 44]


def augment(img: np.ndarray) -> np.ndarray:
    h, w = img.shape


    angle = np.random.uniform(-8, 8)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

    sx = np.random.uniform(0.85, 1.15)
    sy = np.random.uniform(0.85, 1.15)
    M_scale = np.float32([[sx, 0, w * (1 - sx) / 2], [0, sy, h * (1 - sy) / 2]])
    img = cv2.warpAffine(img, M_scale, (w, h), borderMode=cv2.BORDER_REPLICATE)
    

    if np.random.rand() > 0.5:
        noise_mask = np.random.rand(*img.shape)
        img[noise_mask < 0.05] = 0   
        img[noise_mask > 0.95] = 255 
        
 
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


    if np.random.rand() > 0.5:
        k_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k_size, k_size), 0)


    if np.random.rand() > 0.5:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        if np.random.rand() > 0.5:
            img = cv2.erode(img, kernel, iterations=1) 
        else:
            img = cv2.dilate(img, kernel, iterations=1) 


    brightness = np.random.randint(-30, 30)
    img = np.clip(img.astype(np.int32) + brightness, 0, 255).astype(np.uint8)


    noise = np.random.normal(0, np.random.uniform(3, 12), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img

def normalize_polarity(img: np.ndarray) -> np.ndarray:
    if np.mean(img) < 127:
        img = cv2.bitwise_not(img)
    return img


def get_available_fonts(font_sizes: list[int]) -> list[list]:
    """Betölti az összes .ttf és .otf betűtípust a data/fonts/ mappából és annak almappáiból."""
    font_dir = Path("data/fonts")
    available_fonts = []
    
    if not font_dir.exists():
        print(f"[FIGYELMEZTETÉS] A {font_dir} mappa nem létezik, létrehozom...")
        font_dir.mkdir(parents=True, exist_ok=True)
        return [[ImageFont.load_default()] * len(font_sizes)]

    font_paths = []
    for ext in ["*.[tT][tT][fF]", "*.[oO][tT][fF]"]:
        font_paths.extend(list(font_dir.rglob(ext)))

    if not font_paths:
        print("[FIGYELMEZTETÉS] Nem találtam .ttf vagy .otf fájlokat a data/fonts/ mappában!")
        return [[ImageFont.load_default()] * len(font_sizes)]

    print(f"\n[DIAGNOSZTIKA] A data/fonts/ mappában talált font fájlok száma: {len(font_paths)}")
    for p in font_paths:
        print(f"   -> Fájl megtalálva: {p.relative_to(font_dir)}")

    print("\n[INFO] Betűtípusok betöltése és ellenőrzése...")
    for path in font_paths:
        sizes_for_font = []
        font_loaded_successfully = True
        
        for size in font_sizes:
            try:
                font = ImageFont.truetype(str(path), size)
                sizes_for_font.append(font)
            except (IOError, OSError) as e:
                print(f"   [SIKERTELEN] {path.name} ({size}px) - Nem sikerült beolvasni. Hiba: {e}")
                font_loaded_successfully = False
                break 
        
        if font_loaded_successfully and sizes_for_font:
            print(f"   [SIKERES] {path.name} betöltve és használatba véve.")
            available_fonts.append(sizes_for_font)
            
    if not available_fonts:
        print("[FIGYELMEZTETÉS] Egyetlen betűtípust sem sikerült érvényesen betölteni! Default fontot használunk.")
        return [[ImageFont.load_default()] * len(font_sizes)]
        
    print(f"[INFO] Tanításhoz sikeresen előkészített betűtípusok száma: {len(available_fonts)}\n")
    return available_fonts

def generate_char_image_with_font(char: str, font) -> np.ndarray:
    """Egyetlen karakterképet generál, szorosan körbevágja, majd arányosan paddeli."""

    temp_size = 100
    img = Image.new("L", (temp_size, temp_size), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), char, fill=0, font=font)
    arr = np.array(img, dtype=np.uint8)


    coords = cv2.findNonZero(255 - arr) 
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = arr[y:y+h, x:x+w]
    else:
        cropped = arr


    final_img = resize_with_pad(cropped, (IMG_WIDTH, IMG_HEIGHT), bg_color=255)
    
    return final_img

def generate_dataset() -> tuple[list[np.ndarray], list[str]]:
    images, labels = [], []
    
    available_fonts = get_available_fonts(FONT_SIZES)
    augmentations_per_base = 5 
    
    total_samples = len(CHARACTERS) * len(available_fonts) * len(FONT_SIZES) * augmentations_per_base
    
    print(f"[INFO] Szintetikus adathalmaz generálása (minden betűtípussal)...")
    print(f"       {len(CHARACTERS)} karakter × {len(available_fonts)} betűtípus × {len(FONT_SIZES)} méret × {augmentations_per_base} variáció = {total_samples} kép\n")

    for char in CHARACTERS:
        for font_group in available_fonts:
            for font in font_group:
                base_img = generate_char_image_with_font(char, font)
                

                images.append(normalize_polarity(base_img.copy()))
                labels.append(char)
                

                for _ in range(augmentations_per_base - 1):
                    aug_img = augment(base_img.copy())
                    aug_img = normalize_polarity(aug_img)
                    images.append(aug_img)
                    labels.append(char)

    print(f"[INFO] Generálás kész. Összesen {len(images)} szintetikus minta.")
    return images, labels

if __name__ == "__main__":
    all_images, all_labels = generate_dataset()

    print(f"\n[INFO] Teljes adathalmaz: {len(all_images)} tisztán szintetikus minta\n")

    classifier = CharacterClassifier()
    classifier.train(all_images, all_labels, evaluate=True)
    classifier.save()

    print("\n[KÉSZ] Modell elmentve. Futtathatod az app.py-t vagy a main.py-t.")