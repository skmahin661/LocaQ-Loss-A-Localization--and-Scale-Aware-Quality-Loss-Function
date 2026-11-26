# ============================================================
# locaq_loss.py
# VFL + IoU-Conditioned False-Negative Aware Loss (LocaQLoss)
# Compatible with ultralytics 8.4.x AND 8.3.x
# ============================================================
# Novel contribution:
#   fn_weight = 1 + beta * (1 - cls_conf) * (1 - iou_quality)
#
# vs Focal Loss:
#   weight = (1 - pt)^gamma  [cls confidence only]
#
# Usage:
#   from locaq_loss import LocaQLoss
#   import types
#   model = YOLO("yolov8m.yaml")
#   def inject(trainer):
#       trainer.model.init_criterion = types.MethodType(
#           lambda self: LocaQLoss(self, beta=1.0), trainer.model
#       )
#   model.add_callback("on_train_start", inject)
#   model.train(...)
# ============================================================

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou


def _apply_locaq_loss(
    self,
    pred_scores,
    target_scores,
    fg_mask,
    pred_bboxes,
    target_bboxes,
    target_scores_sum,
    dtype,
):
    """
    Core LocaQLoss computation — shared between 8.3.x and 8.4.x.

    Novel formula:
        fn_weight = 1 + beta * (1 - cls_conf) * (1 - iou_quality)

    Returns loss[1] (cls loss) as a scalar tensor.
    """

    # ============================================================
    # FLATTEN — handle both 2D (N, nc) and 3D (batch, anchors, nc)
    # ============================================================

    if pred_scores.ndim == 3:
        _b, _a, _nc = pred_scores.shape
        _ps_flat    = pred_scores.reshape(-1, _nc)
        _pb_flat    = pred_bboxes.reshape(-1, pred_bboxes.shape[-1])
        _tgt_flat   = target_scores.to(dtype).reshape(-1, _nc)
        _tb_flat    = target_bboxes.reshape(-1, target_bboxes.shape[-1])
    else:
        _ps_flat    = pred_scores
        _pb_flat    = pred_bboxes
        _tgt_flat   = target_scores.to(dtype)
        _tb_flat    = target_bboxes

    _pred_sigmoid = _ps_flat.sigmoid()

    # ============================================================
    # STEP 1 — VARIFOCAL BASE LOSS
    # FG: quality-weighted by soft IoU target
    # BG: focal-suppressed by alpha * p^gamma
    # ============================================================

    _vfl_weight = torch.where(
        _tgt_flat > 0,
        _tgt_flat,
        self.vfl_alpha * _pred_sigmoid.pow(self.vfl_gamma)
    )

    _cls_raw  = F.binary_cross_entropy_with_logits(
        _ps_flat, _tgt_flat, reduction="none"
    )

    _vfl_loss = _vfl_weight * _cls_raw

    # ============================================================
    # STEP 2 — IoU-CONDITIONED FN-AWARE WEIGHT
    # fn_weight = 1 + beta * (1 - cls_conf) * (1 - iou_quality)
    # ============================================================

    _fn_weight_map = torch.ones(
        _ps_flat.shape[0],
        device=self.device,
        dtype=_ps_flat.dtype
    )

    _fg_mask_bool = (
        fg_mask.squeeze(-1) if fg_mask.ndim == 3 else fg_mask
    ).flatten()

    if _fg_mask_bool.sum() > 10:

        with torch.no_grad():

            # Classification confidence
            _cls_conf = _pred_sigmoid[
                _fg_mask_bool
            ].max(dim=1)[0].clamp(0.0, 1.0)

            # Localization quality via CIoU
            _fg_pred_boxes   = _pb_flat[_fg_mask_bool]
            _fg_target_boxes = _tb_flat[_fg_mask_bool]

            _iou_quality = bbox_iou(
                _fg_pred_boxes,
                _fg_target_boxes,
                xywh=False,
                CIoU=True
            ).squeeze(-1).clamp(0.0, 1.0)

            # Size-aware weight
            _bw   = (_fg_target_boxes[:, 2] - _fg_target_boxes[:, 0]).clamp(min=1e-6)
            _bh   = (_fg_target_boxes[:, 3] - _fg_target_boxes[:, 1]).clamp(min=1e-6)
            _area = (_bw * _bh).clamp(min=1e-6)

            _size_w = torch.sqrt(
                torch.tensor(
                    self.anchor_area,
                    device=_area.device,
                    dtype=_area.dtype
                ) / _area
            ).clamp(1.0, 2.5)

            # IoU-conditioned FN weight
            _fn_w = (
                1.0
                + self.beta
                * (1.0 - _cls_conf)
                * (1.0 - _iou_quality)
            ) * _size_w

            _fn_w = _fn_w.clamp(
                1.0, self.fn_clamp_max
            ).to(_fn_weight_map.dtype)

        _fn_weight_map[_fg_mask_bool] = _fn_w

    # ============================================================
    # STEP 3 — COMBINE AND NORMALIZE
    # ============================================================

    _fn_weight_expanded = (
        _fn_weight_map
        .view(-1, 1)
        .expand_as(_vfl_loss)
        .to(_vfl_loss.dtype)
    )

    _final_loss = _vfl_loss * _fn_weight_expanded

    return _final_loss.sum() / target_scores_sum


class LocaQLoss(v8DetectionLoss):
    """
    Quality-Aware Loss — VFL + IoU-Conditioned FN-Aware Weighting.

    Compatible with ultralytics 8.3.x and 8.4.x.

    Args:
        model         : De-paralleled YOLOv8 model
        beta          : FN penalty strength. Default: 1.0
        vfl_alpha     : VFL alpha for background suppression. Default: 0.75
        vfl_gamma     : VFL gamma for background focal term. Default: 2.0
        anchor_area   : Reference anchor area in px^2. Default: 1024.0
        fn_clamp_max  : Maximum FN weight cap. Default: 3.0

    Example (ultralytics 8.4.x):
        >>> import types
        >>> from ultralytics import YOLO
        >>> from locaq_loss import LocaQLoss
        >>> model = YOLO("yolov8m.yaml")
        >>> def inject(trainer):
        ...     trainer.model.init_criterion = types.MethodType(
        ...         lambda self: LocaQLoss(self, beta=1.0), trainer.model
        ...     )
        >>> model.add_callback("on_train_start", inject)
        >>> model.train(data="data.yaml", epochs=250)
    """

    def __init__(
        self,
        model,
        beta: float         = 1.0,
        vfl_alpha: float    = 0.75,
        vfl_gamma: float    = 2.0,
        anchor_area: float  = 1024.0,
        fn_clamp_max: float = 3.0,
    ):
        super().__init__(model)
        self.beta         = beta
        self.vfl_alpha    = vfl_alpha
        self.vfl_gamma    = vfl_gamma
        self.anchor_area  = anchor_area
        self.fn_clamp_max = fn_clamp_max

    def get_assigned_targets_and_loss(self, preds, batch):
        """
        Override for ultralytics 8.4.x.
        Replaces BCE cls loss with VFL + IoU-conditioned FN-aware loss.
        """
        loss = torch.zeros(3, device=self.device)

        from ultralytics.utils.tal import make_anchors

        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype      = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz      = torch.tensor(
            preds["feats"][0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets              = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt              = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Task-aligned assignment
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ============================================================
        # VFL + IoU-CONDITIONED FN-AWARE CLS LOSS
        # ============================================================

        loss[1] = _apply_locaq_loss(
            self,
            pred_scores,
            target_scores,
            fg_mask,
            pred_bboxes,
            target_bboxes,
            target_scores_sum,
            dtype,
        )

        # Box + DFL loss — unchanged
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )

    def __call__(self, preds, batch):
        """
        Override for ultralytics 8.3.x (older structure).
        Falls back to parent for 8.4.x (which uses get_assigned_targets_and_loss).
        """
        # 8.4.x: preds is a dict — use parent __call__ which routes to
        # get_assigned_targets_and_loss (our override above)
        if isinstance(preds, dict) or (
            isinstance(preds, tuple) and not isinstance(preds[1], (list, tuple))
        ):
            return super().__call__(preds, batch)

        # 8.3.x: preds is a tuple with list of feature tensors
        loss       = torch.zeros(3, device=self.device)
        feats      = preds[1] if isinstance(preds, tuple) else preds
        batch_size = feats[0].shape[0]

        pred_distri, pred_scores = torch.cat(
            [xi.view(batch_size, self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(
            feats[0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]

        from ultralytics.utils.tal import make_anchors
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets              = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt              = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # VFL + IoU-Conditioned FN-Aware cls loss
        loss[1] = _apply_locaq_loss(
            self,
            pred_scores,
            target_scores,
            fg_mask,
            pred_bboxes,
            target_bboxes,
            target_scores_sum,
            dtype,
        )

        # Box + DFL loss — unchanged
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss * batch_size, loss.detach()
