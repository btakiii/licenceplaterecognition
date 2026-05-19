import streamlit as st
import cv2
import numpy as np
from PIL import Image


from src.detection.plate_detector import PlateDetector
from src.preprocessing.image_pipeline import preprocess
from src.segmentation.char_segmenter import segment_characters
from src.recognition.knn_classifier import CharacterClassifier


st.set_page_config(page_title="Rendszámfelismerő", layout="wide")
st.title("🚗 Automatikus Rendszámtábla Felismerő (ALPR)")


@st.cache_resource
def load_models():
    detector = PlateDetector()
    classifier = CharacterClassifier()
    return detector, classifier

detector, classifier = load_models()


uploaded_file = st.file_uploader("Tölts fel egy képet az autóról", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Feldolgozás eredményei")
    
    detections = detector.detect(image_bgr)
    
    if not detections:
        st.warning("Nem sikerült rendszámtáblát találni a képen.")
        st.image(image_rgb, caption="Eredeti kép", use_column_width=True)
    else:
        for idx, det in enumerate(detections):
            roi_bgr = det['roi']
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            conf = det['confidence']
            
            st.markdown(f"### Rendszámtábla #{idx+1} (Megbízhatóság: {conf:.1%})")

            binary, steps = preprocess(roi_bgr, visualize=True)
            char_candidates, seg_debug = segment_characters(binary, visualize=True)

            plate_text = ""
            if char_candidates and classifier.svm is not None:
                char_images = [img for (_, img) in char_candidates]
                predicted = classifier.predict_batch(char_images)
                plate_text = "".join(predicted)
            else:
                plate_text = "ISMERETLEN"

            tab1, tab2, tab3, tab4 = st.tabs(["Végeredmény", "Előfeldolgozás Lépései", "Szegmentálás", "Karakterek"])
            
            with tab1:
                st.success(f"Felismert rendszám: **{plate_text}**")
                res_img = image_rgb.copy()
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(res_img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                st.image(res_img, width='stretch')

            with tab2:
                st.write("A képtisztítás és binarizálás lépései egymás alatt:")
                st.image(roi_rgb, caption="Eredeti ROI", width='stretch')
                for step_name, step_img in steps.items():
                    cmap = "RGB" if len(step_img.shape) == 3 else "gray"
                    if cmap == "RGB":
                        step_img = cv2.cvtColor(step_img, cv2.COLOR_BGR2RGB)
                    
                    st.image(step_img, caption=step_name, width='stretch', channels=cmap if cmap == "RGB" else "GRAY")

            with tab3:
                st.write("A megtalált karakter-kontúrok (zölddel az elfogadott):")
                st.image(cv2.cvtColor(seg_debug['annotated'], cv2.COLOR_BGR2RGB), width='stretch')

            with tab4:
                st.write("A kivágott és RF-nek átadott karakterek:")
                if char_candidates:
                    for i, ((_, char_img), pred) in enumerate(zip(char_candidates, predicted)):
                        st.image(char_img, caption=f"Pred: {pred}", width=200, channels="GRAY")
                else:
                    st.info("Nem talált karaktert a szegmentáló.")
            
            st.markdown("---")
