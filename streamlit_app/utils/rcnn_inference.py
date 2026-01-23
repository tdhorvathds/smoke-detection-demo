from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T

from streamlit_app.src.rcnn_model import RCNNConfig, get_fasterrcnn_model
from streamlit_app.utils.weights import ensure_weights


BEST_RCNN_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "fasterrcnn_best.pth"

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


_tf = T.ToTensor()

@lru_cache(maxsize=1)
def load_best_rcnn():
    weights = ensure_weights()
    w_path = weights.get("fasterrcnn_best.pth", BEST_RCNN_WEIGHTS)

    cfg = RCNNConfig(pretrained=True, num_classes=2, label_smoothing=0.1)
    model = get_fasterrcnn_model(cfg)

    state = torch.load(w_path, map_location="cpu")
    state = state["model"] if isinstance(state, dict) and "model" in state else state

    model.load_state_dict(state, strict=True)
    model.to(_device())
    model.eval()
    return model

def predict_rcnn(pil_image: Image.Image, conf: float = 0.25):
    model = load_best_rcnn()
    img_t = _tf(pil_image.convert("RGB")).to(_device())

    with torch.no_grad():
        out = model([img_t])[0]

    boxes = out["boxes"].detach().cpu()
    scores = out["scores"].detach().cpu()
    labels = out["labels"].detach().cpu()

    dets = []
    for xyxy, s, lbl in zip(boxes, scores, labels):
        if float(s) < conf:
            continue
        dets.append(
            {
                "xyxy": [float(x) for x in xyxy.tolist()],
                "score": float(s),
                "cls_id": int(lbl.item()),
            }
        )
    return dets
