from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


@dataclass(frozen=True)
class RCNNConfig:
    # model init
    pretrained: bool = True
    num_classes: int = 2  # background + smoke

    # training-time choice that must match weights (loss definition)
    label_smoothing: float = 0.1

    # RPN / anchors (these affect architecture + proposals)
    anchor_sizes: Tuple[Tuple[int, ...], ...] = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios: Tuple[Tuple[float, ...], ...] = (
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
    )
    rpn_fg_iou_thresh: float = 0.4
    rpn_bg_iou_thresh: float = 0.1

    rpn_pre_nms_top_n_train: int = 4000
    rpn_post_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 2000
    rpn_post_nms_top_n_test: int = 1000

    detections_per_img: int = 200

    # debug printing (off by default in a recruiter repo)
    verbose: bool = False


def get_fasterrcnn_model(cfg: RCNNConfig) -> nn.Module:
    """
    Build Faster R-CNN ResNet50-FPNv2 and patch:
      - anchors for smaller objects
      - relaxed RPN IoU thresholds
      - proposal counts
      - ROI head predictor (num_classes)
      - optional label smoothing in classification loss
    """
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if cfg.pretrained else None,
        weights_backbone=None,
    )

    # Anchors
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=cfg.anchor_sizes,
        aspect_ratios=cfg.aspect_ratios,
    )

    # RPN matching thresholds + proposal capacity
    model.rpn.fg_iou_thresh = cfg.rpn_fg_iou_thresh
    model.rpn.bg_iou_thresh = cfg.rpn_bg_iou_thresh

    model.rpn.pre_nms_top_n_train = cfg.rpn_pre_nms_top_n_train
    model.rpn.post_nms_top_n_train = cfg.rpn_post_nms_top_n_train
    model.rpn.pre_nms_top_n_test = cfg.rpn_pre_nms_top_n_test
    model.rpn.post_nms_top_n_test = cfg.rpn_post_nms_top_n_test

    # ROI head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)
    model.roi_heads.detections_per_img = cfg.detections_per_img

    # Classification loss (must match training if you used label smoothing)
    if cfg.label_smoothing and cfg.label_smoothing > 0:
        model.roi_heads.fastrcnn_loss_func = torch.nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing
        )
    else:
        model.roi_heads.fastrcnn_loss_func = torch.nn.CrossEntropyLoss()

    if cfg.verbose:
        print("[RPN] Anchor sizes:", model.rpn.anchor_generator.sizes)
        print("[RPN] FG/BG IoU:", model.rpn.fg_iou_thresh, model.rpn.bg_iou_thresh)

    return model
