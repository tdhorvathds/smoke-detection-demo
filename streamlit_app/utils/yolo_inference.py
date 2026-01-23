from functools import lru_cache
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from streamlit_app.utils.weights import ensure_weights

BEST_YOLO_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "yolo_best.pt"

@lru_cache(maxsize=1)
def load_best_yolo():
    weights = ensure_weights()
    w_path = weights.get("yolo_best.pt", BEST_YOLO_WEIGHTS)

    model = YOLO(str(w_path))
    model.fuse()  # optional

    model.to("cpu")  # keep for Streamlit Cloud; can remove later if you deploy with GPU
    return model

def predict_yolo(pil_image: Image.Image, conf: float = 0.25, iou: float = 0.7):
    """
    Run YOLO on a single PIL image and return a list of detections.
    Each detection: {'xyxy': [x1, y1, x2, y2], 'score': float, 'cls_id': int}
    """
    model = load_best_yolo()
    results = model(pil_image, conf=conf, iou=iou, verbose=False)[0]

    boxes = []
    if results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            boxes.append({"xyxy": xyxy, "score": score, "cls_id": cls_id})
    return boxes
