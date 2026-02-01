"""
Streamlit demo: Wildfire Smoke Detection (YOLOv11s vs Faster R-CNN)

"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

for p in [str(REPO_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


from streamlit_app.utils.assets_download import ensure_demo_assets
from streamlit_app.utils.yolo_inference import load_best_yolo, predict_yolo
from streamlit_app.utils.rcnn_inference import load_best_rcnn, predict_rcnn


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------

st.markdown(
    """
    <style>
      /* Always reserve vertical scrollbar space to prevent layout shift */
      html, body {
        overflow-y: scroll;
      }

      /* Your existing wide-layout overrides */
      .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
      }
      section.main > div {
        max-width: 100% !important;
      }
      div[data-testid="stAppViewContainer"] {
        max-width: 100% !important;
      }
      div[data-testid="stMainBlockContainer"] {
        max-width: 100% !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


ensure_demo_assets()

# ------------------------------------------------------------
# Folder structure (EXPECTED)
# ------------------------------------------------------------
# Curated sample structure:
# assets/samples_demo/
#   yolo/clean/<scene_key>.jpg
#   yolo/noise/<scene_key>.jpg
#   rcnn/clean/<scene_key>.jpg
#   rcnn/fog/<scene_key>.jpg
SAMPLES_ROOT = APP_DIR / "assets" / "samples_demo"

# Curated XAI overlay structure:
# assets/xai_demo/
#   yolo/{gradcampp,drise}/{clean,noise}/<scene_key>.png
#   rcnn/{gradcampp,drise}/{clean,fog}/<scene_key>.png
XAI_ROOT = APP_DIR / "assets" / "xai_demo"

# OPTIONAL: COCO JSON for GT boxes for sample images only
GT_COCO_JSON = APP_DIR / "assets" / "gt" / "demo_subset.json"

MODEL_DISPLAY_NAME = {
    "yolo": "YOLOv11s",
    "rcnn": "Faster R-CNN",
}

METHOD_DISPLAY_NAME = {
    "gradcampp": "Grad-CAM++",
    "drise": "D-RISE",
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def render_box_legend(show_gt: bool, show_pred: bool):
    items = []
    if show_gt:
        items.append("🟩 Ground Truth (GT)")
    if show_pred:
        items.append("🟥 Predicted")
    if items:
        st.markdown(
            "**Box legend:** " + " &nbsp;&nbsp; ".join(items),
            help="Color coding of bounding boxes"
        )


def _list_images_by_stem(folder: Path) -> Dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png"}
    out: Dict[str, Path] = {}
    if not folder.exists():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out.setdefault(p.stem, p)  # first one wins
    return out


def load_samples_by_arch(samples_root: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """
    Returns:
      samples[arch][scene_key][variant] = Path

    - scene_key is the STEM (no extension).
    - clean entries are included ONLY if the required corrupted counterpart exists:
        YOLO: clean + noise
        RCNN: clean + fog
    """
    samples = {"yolo": {}, "rcnn": {}}

    # YOLO: clean + noise
    y_clean = _list_images_by_stem(samples_root / "yolo" / "clean")
    y_noise = _list_images_by_stem(samples_root / "yolo" / "noise")
    for stem, p_clean in y_clean.items():
        p_noise = y_noise.get(stem)
        if p_noise is None:
            continue
        samples["yolo"][stem] = {"clean": p_clean, "noise": p_noise}

    # RCNN: clean + fog
    r_clean = _list_images_by_stem(samples_root / "rcnn" / "clean")
    r_fog = _list_images_by_stem(samples_root / "rcnn" / "fog")
    for stem, p_clean in r_clean.items():
        p_fog = r_fog.get(stem)
        if p_fog is None:
            continue
        samples["rcnn"][stem] = {"clean": p_clean, "fog": p_fog}

    return samples


def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def top_k_by_score(boxes: List[dict], k: int) -> List[dict]:
    if k <= 0:
        return boxes
    return sorted(boxes, key=lambda d: float(d.get("score", 0.0)), reverse=True)[:k]


def detections_to_df(boxes: List[dict]) -> pd.DataFrame:
    rows = []
    for i, b in enumerate(boxes, start=1):
        x1, y1, x2, y2 = b["xyxy"]
        rows.append(
            {
                "id": f"D{i}",
                "score": float(b.get("score", 0.0)),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cls_id": int(b.get("cls_id", -1)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False, ignore_index=True)
    return df


def draw_pred_boxes(img: Image.Image, boxes: List[dict], color: str = "red") -> Image.Image:
    out = img.convert("RGB").copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)

    # Box + label style
    box_width = 5
    text_pad = 3
    font_color = "white"
    label_bg = color  # same color as box

    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except IOError:
        font = ImageFont.load_default()

    for idx, b in enumerate(boxes, start=1):
        x1, y1, x2, y2 = b["xyxy"]
        x1 = clamp_int(x1, 0, w - 1)
        y1 = clamp_int(y1, 0, h - 1)
        x2 = clamp_int(x2, 0, w - 1)
        y2 = clamp_int(y2, 0, h - 1)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        score = b.get("score", None)

        label = f"P{idx} {float(score):.2f}" if score is not None else f"P{idx}"

        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        tx1 = x1
        ty1 = max(y1 - text_h - 2 * text_pad, 0)
        tx2 = min(tx1 + text_w + 2 * text_pad, w)
        ty2 = ty1 + text_h + 2 * text_pad

        draw.rectangle([tx1, ty1, tx2, ty2], fill=label_bg)
        draw.text(
            (tx1 + text_pad, ty1 + text_pad),
            label,
            fill=font_color,
            font=font,
        )

    return out


def draw_gt_boxes(img: Image.Image, gt_xyxy: List[List[float]], color: str = "lime") -> Image.Image:
    out = img.convert("RGB").copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)

    for idx, (x1, y1, x2, y2) in enumerate(gt_xyxy, start=1):
        x1 = clamp_int(x1, 0, w - 1)
        y1 = clamp_int(y1, 0, h - 1)
        x2 = clamp_int(x2, 0, w - 1)
        y2 = clamp_int(y2, 0, h - 1)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"GT{idx}"
        text_y = max(y1 - 12, 0)
        draw.text((x1, text_y), label, fill=color)

    return out


def alpha_blend(base_img: Image.Image, overlay_img: Image.Image, alpha: float) -> Image.Image:
    base = base_img.convert("RGBA")
    ov = overlay_img.convert("RGBA")
    if ov.size != base.size:
        ov = ov.resize(base.size)

    ov_arr = np.array(ov).astype(np.float32)
    ov_arr[..., 3] = ov_arr[..., 3] * float(alpha)
    ov = Image.fromarray(np.clip(ov_arr, 0, 255).astype(np.uint8), mode="RGBA")

    return Image.alpha_composite(base, ov).convert("RGB")


@st.cache_resource
def get_yolo_model():
    return load_best_yolo()


@st.cache_resource
def get_rcnn_model():
    return load_best_rcnn()


def find_xai_overlay(model_key: str, method: str, variant: str, scene_key: str) -> Optional[Path]:
    """
    scene_key is a stem (no extension), e.g. "TP_002753" or "002753".
    Looks for common extensions under:
      XAI_ROOT / model_key / method / variant / <scene_key>.<ext>
    """
    base = XAI_ROOT / model_key / method / variant
    if not base.exists():
        return None

    candidates = [
        base / f"{scene_key}.png",
        base / f"{scene_key}.jpg",
        base / f"{scene_key}.jpeg",
        ]
    for c in candidates:
        if c.exists():
            return c
    return None


@st.cache_data
def load_gt_by_key() -> Dict[str, List[List[float]]]:
    """
    Returns: dict[scene_key] -> list of GT boxes (xyxy)
    scene_key = Path(file_name).stem
    """
    if not GT_COCO_JSON.exists():
        return {}

    import json
    from pathlib import Path

    with open(GT_COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    id_to_key = {}
    for im in images:
        fname = Path(str(im.get("file_name", ""))).name
        if "id" in im and fname:
            scene_key = Path(fname).stem
            id_to_key[int(im["id"])] = scene_key

    out: Dict[str, List[List[float]]] = {}
    for ann in anns:
        img_id = ann.get("image_id", None)
        bbox = ann.get("bbox", None)
        if img_id is None or bbox is None:
            continue

        scene_key = id_to_key.get(int(img_id), None)
        if not scene_key:
            continue

        x, y, w, h = bbox
        xyxy = [float(x), float(y), float(x + w), float(y + h)]
        out.setdefault(scene_key, []).append(xyxy)

    return out


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
def main():
    st.title("Wildfire Smoke Detection Demo")

    st.markdown(
        "In-domain samples with precomputed XAI overlays. "
        "YOLO is shown on **clean + noise**, Faster R-CNN on **clean + fog**."
    )


    # Load curated samples
    samples_by_arch = load_samples_by_arch(SAMPLES_ROOT)

    # GT mapping
    gt_map = load_gt_by_key()

    tab_detect, tab_explain = st.tabs(["Detect", "Explain"])

    # =========================
    # Detect tab
    # =========================
    with tab_detect:
        col_left, col_right = st.columns([1, 3], gap="large")

        with col_left:
            st.subheader("Input")

            model_choice = st.radio("Model", ["YOLOv11s", "Faster R-CNN"], horizontal=True)
            source = st.radio("Image source", ["Sample", "Upload"], horizontal=True)

            img: Optional[Image.Image] = None
            chosen_key: Optional[str] = None
            variant: str = "clean"

            arch_key = "yolo" if model_choice.startswith("YOLO") else "rcnn"
            arch_samples = samples_by_arch.get(arch_key, {})

            if source == "Sample":
                if not arch_samples:
                    st.warning(
                        f"No curated paired samples found for {arch_key.upper()}.\n\n"
                        f"Expected folders under: {SAMPLES_ROOT}\n"
                        f"- yolo/clean + yolo/noise\n"
                        f"- rcnn/clean + rcnn/fog"
                    )
                else:
                    chosen_key = st.selectbox("Choose scene", sorted(arch_samples.keys()))
                    # keep stable order
                    variants = [v for v in ["clean", "noise", "fog"] if v in arch_samples[chosen_key]]
                    variant = st.radio("Variant", variants, horizontal=True)
                    img = Image.open(arch_samples[chosen_key][variant]).convert("RGB")
            else:
                uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
                if uploaded:
                    img = Image.open(uploaded).convert("RGB")

            conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
            yolo_iou = 0.70
            if model_choice.startswith("YOLO"):
                yolo_iou = st.slider("YOLO IoU (NMS)", 0.30, 0.95, 0.70, 0.05)

            max_det = st.slider("Max detections (top-k by score)", 1, 30, 10, 1)

            show_pred = st.checkbox("Show predicted boxes", value=True, key="det_show_pred")
            show_gt = st.checkbox("Show GT boxes (if available)", value=False, key="det_show_gt")

            run = st.button("Run detection", type="primary", use_container_width=True)


            with st.expander("Notes"):
                st.write("- Sample list is architecture-specific and only shows **clean scenes that have the required corrupted counterpart**.")
                st.write("- GT boxes are only available for sample images if you provide a COCO JSON in assets/gt/.")
                st.write(f"- GT JSON found: {'YES' if GT_COCO_JSON.exists() else 'NO'}")
                st.write("- For uploaded images, GT is not available.")

        with col_right:
            st.subheader("Output")

            # Stable placeholders (prevent layout collapse/expand on reruns)
            msg_slot = st.empty()
            img_slot = st.empty()
            table_slot = st.empty()

            # Clear everything by default on each rerun
            msg_slot.empty()
            img_slot.empty()
            table_slot.empty()

            if not run:
                msg_slot.info("Choose an image and click **Run detection**.")
            else:
                if img is None:
                    msg_slot.warning("No image selected/uploaded.")
                else:
                    try:
                        with st.spinner("Running inference..."):
                            if model_choice.startswith("YOLO"):
                                _ = get_yolo_model()
                                boxes = predict_yolo(img, conf=conf, iou=float(yolo_iou))
                                model_label = "YOLOv11s"
                            else:
                                _ = get_rcnn_model()
                                boxes = predict_rcnn(img, conf=conf)
                                model_label = "Faster R-CNN"

                        boxes = top_k_by_score(boxes, k=max_det)

                        out_img = img

                        # Draw GT first
                        if show_gt and source == "Sample" and chosen_key and chosen_key in gt_map:
                            out_img = draw_gt_boxes(out_img, gt_map[chosen_key], color="lime")
                        elif show_gt and source == "Sample" and chosen_key and (chosen_key not in gt_map):
                            msg_slot.caption("GT boxes requested, but none found for this image in the provided COCO JSON.")

                        # Draw predictions
                        if show_pred and boxes:
                            out_img = draw_pred_boxes(out_img, boxes, color="red")

                        render_box_legend(show_gt=show_gt, show_pred=show_pred)

                        # Image output (kept stable via placeholder)
                        img_slot.image(
                            out_img,
                            caption=f"{model_label} • {variant}",
                            use_container_width=True,
                        )

                        # Table output (fixed height reduces HF jitter a lot)
                        if boxes:
                            df = detections_to_df(boxes)
                            table_slot.dataframe(
                                df,
                                use_container_width=True,
                                hide_index=True,
                                height=320,
                            )
                        else:
                            msg_slot.info("No smoke detections above the confidence threshold.")

                    except Exception as e:
                        msg_slot.error("Inference failed.")
                        # keep exception under the message area; still stable structure
                        msg_slot.exception(e)


    # =========================
    # Explain tab
    # =========================
    with tab_explain:
        st.subheader("Explain")
        st.caption("Shows precomputed Grad-CAM++ / D-RISE overlays. No live XAI computation.")

        col_l, col_r = st.columns([1, 3], gap="large")

        with col_l:
            model_label = st.radio(
                "Model overlays",
                ["YOLOv11s", "Faster R-CNN"],
                horizontal=True,
            )

            model_key = "yolo" if model_label == "YOLOv11s" else "rcnn"

            method_label = st.radio(
                "Method",
                ["Grad-CAM++", "D-RISE"],
                horizontal=True,
            )

            method = "gradcampp" if method_label == "Grad-CAM++" else "drise"

            expl_samples = samples_by_arch.get(model_key, {})
            if not expl_samples:
                st.warning(f"No curated paired samples found for {model_key.upper()} under: {SAMPLES_ROOT}")
                st.stop()

            chosen_key = st.selectbox("Choose scene", sorted(expl_samples.keys()), key="expl_key")

            variants = [v for v in (["clean", "noise"] if model_key == "yolo" else ["clean", "fog"]) if v in expl_samples[chosen_key]]
            if not variants:
                st.warning("No matching variants for this scene in your samples folder.")
                st.stop()

            variant = st.radio("Variant", variants, horizontal=True, key="expl_variant")

            display_mode = st.radio(
                "Display",
                ["Overlay only", "Base + overlay"],
                horizontal=False,
            )
            overlay_alpha = st.slider("Overlay opacity", 0.10, 1.00, 0.55, 0.05)

            show_pred_expl = st.checkbox("Show predicted boxes", value=False, key="expl_show_pred")
            show_gt_expl = st.checkbox("Show GT boxes (if available)", value=False, key="expl_show_gt")

            box_conf = st.slider("Box confidence", 0.05, 0.95, 0.25, 0.05)
            yolo_iou_expl = 0.70
            if model_key == "yolo":
                yolo_iou_expl = st.slider("YOLO IoU", 0.30, 0.95, 0.70, 0.05)

        with col_r:
            expl_boxes = None
            base_path = expl_samples[chosen_key][variant]
            base_img = Image.open(base_path).convert("RGB")

            overlay_path = find_xai_overlay(model_key=model_key, method=method, variant=variant, scene_key=chosen_key)

            if overlay_path is None:
                st.info(
                    "No overlay found for this selection.\n\nExpected, for example:\n"
                    f"- {XAI_ROOT / model_key / method / variant / (chosen_key + '.png')}"
                )
                st.image(base_img, caption="Base image", use_container_width=True)
                st.stop()

            overlay_img = Image.open(overlay_path).convert("RGB")

            if display_mode == "Overlay only":
                disp = overlay_img
            else:
                disp = alpha_blend(base_img, overlay_img, alpha=overlay_alpha)

            # Optional GT
            if show_gt_expl and chosen_key in gt_map:
                disp = draw_gt_boxes(disp, gt_map[chosen_key], color="lime")
            elif show_gt_expl and chosen_key not in gt_map:
                st.caption("GT boxes requested, but none found for this image in the provided COCO JSON.")

            # Optional predicted boxes
            if show_pred_expl:
                try:
                    with st.spinner("Computing predicted boxes..."):
                        if model_key == "yolo":
                            _ = get_yolo_model()
                            boxes = predict_yolo(
                                base_img,
                                conf=box_conf,
                                iou=float(yolo_iou_expl),
                            )
                        else:
                            _ = get_rcnn_model()
                            boxes = predict_rcnn(base_img, conf=box_conf)

                    boxes = top_k_by_score(boxes, k=30)

                    if boxes:
                        disp = draw_pred_boxes(disp, boxes, color="red")
                        expl_boxes = boxes
                    else:
                        st.caption("No predicted boxes above the selected confidence.")

                except Exception as e:
                    st.warning("Failed to compute predicted boxes.")
                    st.exception(e)


            render_box_legend(show_gt=show_gt_expl, show_pred=show_pred_expl)

            st.image(
                disp,
                caption=f"{MODEL_DISPLAY_NAME[model_key]} • {METHOD_DISPLAY_NAME[method]} • {variant}",
                use_container_width=True,
            )

            if expl_boxes:
                st.markdown("**Predicted boxes**")
                st.dataframe(
                    detections_to_df(expl_boxes),
                    use_container_width=True,
                    hide_index=True,
                    height=320
                )


if __name__ == "__main__":
    main()
