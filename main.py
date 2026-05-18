"""
main.py
-------
Rendszámtábla felismerő rendszer – fő belépési pont.

Használat:
    python main.py --image data/images/auto.jpg
    python main.py --image data/images/auto.jpg --visualize
    python main.py --image data/images/auto.jpg --no-visualize

Opciók:
    --image       : bemeneti kép elérési útja (kötelező)
    --visualize   : pipeline lépéseinek megjelenítése (alapértelmezett: igen)
    --no-visualize: csak a végeredmény kiírása, vizualizáció nélkül
    --model       : egyedi YOLOv8 modell elérési útja (opcionális)
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

from src.detection.plate_detector      import PlateDetector
from src.preprocessing.image_pipeline  import preprocess
from src.segmentation.char_segmenter   import segment_characters
from src.recognition.knn_classifier    import CharacterClassifier
from src.visualization.visualizer      import show_pipeline, show_result


def recognize_plate(image: np.ndarray,
                    detector: PlateDetector,
                    classifier: CharacterClassifier,
                    visualize: bool = True) -> list[dict]:
    """
    Teljes rendszámfelismerési pipeline futtatása egy képen.

    Paraméterek:
        image      : bemeneti kép (BGR, np.ndarray)
        detector   : inicializált PlateDetector
        classifier : inicializált CharacterClassifier
        visualize  : pipeline lépéseinek megjelenítése

    Visszatér:
        Lista dict-ekkel:
            {
              'bbox'       : (x1, y1, x2, y2),
              'confidence' : float,
              'plate_text' : str,
              'roi'        : np.ndarray
            }
    """

    print("\n[1/4] Rendszámtábla detektálás...")
    detections = detector.detect(image)

    if not detections:
        print("[EREDMÉNY] Nem sikerült rendszámtáblát detektálni.")
        return []

    print(f"       {len(detections)} tábla detektálva.")

    results = []

    for idx, det in enumerate(detections):
        roi  = det['roi']
        bbox = det['bbox']
        conf = det['confidence']

        print(f"\n--- Tábla #{idx + 1} (konfidencia: {conf:.1%}) ---")


        print("[2/4] Képelőfeldolgozás (Deskew + CLAHE + Otsu/Niblack + morfológia)...")
        binary, steps = preprocess(roi, visualize=visualize)


        print("[3/4] Karakterszegmentálás...")
        if visualize:
            char_candidates, seg_debug = segment_characters(binary, visualize=True)
        else:
            char_candidates = segment_characters(binary, visualize=False)
            seg_debug = None

        print(f"       {len(char_candidates)} karakter jelölt találva.")


        print("[4/4] Karakterfelismerés (KNN + HOG)...")

        plate_text = ""
        if char_candidates and classifier.svm is not None:
            char_images = [img for (_, img) in char_candidates]
            predicted   = classifier.predict_batch(char_images)
            plate_text  = "".join(predicted)
        else:
            if classifier.svm is None:
                print("       [FIGYELMEZTETÉS] Nincs betanított modell. "
                      "Futtasd a train_classifier.py-t!")
            plate_text = "ISMERETLEN"

        print(f"\n       *** Felismert rendszám: {plate_text} ***")

        if visualize:
            show_pipeline(roi, steps, char_candidates, plate_text, seg_debug=seg_debug)

        results.append({
            'bbox'       : bbox,
            'confidence' : conf,
            'plate_text' : plate_text,
            'roi'        : roi
        })

    if visualize:
        show_result(image, results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Automatikus rendszámtábla felismerő rendszer"
    )
    parser.add_argument(
        "--image", required=True,
        help="Bemeneti kép elérési útja"
    )
    parser.add_argument(
        "--visualize", action="store_true", default=True,
        help="Pipeline lépéseinek megjelenítése (alapértelmezett)"
    )
    parser.add_argument(
        "--no-visualize", action="store_true",
        help="Vizualizáció kikapcsolása"
    )
    parser.add_argument(
        "--model", default=None,
        help="Egyedi YOLOv8 modell elérési útja (opcionális)"
    )
    args = parser.parse_args()

    visualize = not args.no_visualize


    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[HIBA] A kép nem található: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[HIBA] A kép nem olvasható: {image_path}")
        sys.exit(1)

    print(f"[INFO] Kép betöltve: {image_path} ({image.shape[1]}×{image.shape[0]} px)")

    detector   = PlateDetector(model_path=args.model)
    classifier = CharacterClassifier()

    results = recognize_plate(image, detector, classifier, visualize=visualize)

    print("\n========== ÖSSZEFOGLALÁS ==========")
    if results:
        for i, r in enumerate(results, 1):
            print(f"  Tábla #{i}: {r['plate_text']}  (konfidencia: {r['confidence']:.1%})")
    else:
        print("  Nem sikerült rendszámot felismerni.")
    print("====================================\n")


if __name__ == "__main__":
    main()
