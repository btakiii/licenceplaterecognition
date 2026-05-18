import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_pipeline(original_roi: np.ndarray,
                  steps: dict,
                  char_candidates: list = None,
                  predicted_text: str = None,
                  seg_debug: dict = None):

    preproc_steps = {
        '0_deskew'    : '0. Dőléskorrekció (Hough)',
        '1_gray'      : '1. Szürkeárnyalatos',
        '2_clahe'     : '2. CLAHE kontraszt',
        '3_binary'    : '3. Binarizálás (Otsu/Niblack)',
        '4_morphology': '4. Morfológiai tisztítás',
    }

    visible_keys = [k for k in preproc_steps if k in steps]
    n_cols = len(visible_keys) + 1

    fig1, axes1 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes1 = [axes1]
    fig1.suptitle('Előfeldolgozási pipeline', fontsize=14, fontweight='bold')

    axes1[0].imshow(cv2.cvtColor(original_roi, cv2.COLOR_BGR2RGB))
    axes1[0].set_title('Eredeti ROI')
    axes1[0].axis('off')

    for i, key in enumerate(visible_keys, start=1):
        img   = steps[key]
        label = preproc_steps[key]

        if key == '0_deskew' and len(img.shape) == 3:
            axes1[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes1[i].imshow(img, cmap='gray')

        axes1[i].set_title(label)
        axes1[i].axis('off')

    plt.tight_layout()
    plt.show()

    if seg_debug and 'annotated' in seg_debug:
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
        fig2.suptitle('Karakterszegmentálás – detektált bounding boxok',
                      fontsize=14, fontweight='bold')
        annotated = seg_debug['annotated']
        ax2.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax2.set_title('Zöld = elfogadott   Szürke = elutasított')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    if char_candidates:
        _show_characters(char_candidates, predicted_text)


def _show_characters(char_candidates: list, predicted_text: str = None):
    n = len(char_candidates)
    if n == 0:
        print("[INFO] Nem találhatók karakterek.")
        return

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))
    if n == 1:
        axes = [axes]

    title = 'Szegmentált karakterek'
    if predicted_text:
        title += f'  →  Felismert rendszám: "{predicted_text}"'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for i, (x_pos, char_img) in enumerate(char_candidates):
        axes[i].imshow(char_img, cmap='gray')
        axes[i].set_title(f'#{i+1}', fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def show_result(image: np.ndarray, detections: list):
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        text = det.get('plate_text', '???')
        conf = det.get('confidence', 0.0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)

        label = f"{text}  ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated,
                      (x1, y1 - th - 10), (x1 + tw + 6, y1),
                      (0, 200, 0), -1)
        cv2.putText(annotated, label,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    plt.figure(figsize=(12, 7))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title('Detektált rendszámtáblák', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
