"""YOLOv11 training loss.

Three loss components
─────────────────────
box   CIoU loss on decoded bounding boxes for foreground anchors.
cls   Binary cross-entropy on class logits with soft labels
      (weighted by the TAL alignment score, not hard 0/1).
dfl   Cross-entropy on the coordinate distribution.  The target is a
      soft two-bin distribution centred on the true distance value.

TaskAlignedAssigner
───────────────────
For each ground-truth box we compute a per-anchor alignment score:

    t = score(pred_cls for gt_class) ** alpha  *  IoU(pred_box, gt_box) ** beta

and assign the top-k anchors per GT.  If an anchor is claimed by multiple
GTs, the one with the higher score wins.

References
──────────
TAL  – TOOD: Task-aligned One-stage Object Detection (Feng et al., 2021)
DFL  – Generalized Focal Loss (Li et al., 2020)
CIoU – Distance-IoU Loss (Zheng et al., 2019)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_anchors(
    feats: list[Tensor],
    strides: Tensor,
    offset: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Build anchor center points for all feature scales.

    Args:
        feats:   list of [B, C, Hi, Wi] — used only for H/W shape.
        strides: 1-D tensor, e.g. tensor([8., 16., 32.]).
        offset:  fraction of a grid cell to add (0.5 = cell centre).

    Returns:
        anchor_points: [N, 2]   xy pixel coordinates.
        stride_tensor: [N, 1]   stride repeated for each anchor.
    """
    anc, strd = [], []
    for feat, s in zip(feats, strides):
        h, w = feat.shape[2:]
        xs = torch.arange(w, device=feat.device, dtype=torch.float32) + offset
        ys = torch.arange(h, device=feat.device, dtype=torch.float32) + offset
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([gx, gy], -1).reshape(-1, 2) * s
        anc.append(pts)
        strd.append(pts.new_full((pts.shape[0], 1), s))
    return torch.cat(anc), torch.cat(strd)


def dist2bbox(dist: Tensor, anchor_pts: Tensor) -> Tensor:
    """lt/rb offsets + anchor centre → xyxy box (same units as inputs).

    dist:        [..., 4]  (left, top, right, bottom) offsets
    anchor_pts:  [..., 2]  (cx, cy)
    """
    lt, rb = dist.chunk(2, dim=-1)
    return torch.cat([anchor_pts - lt, anchor_pts + rb], dim=-1)


def bbox2dist(bboxes: Tensor, anchor_pts: Tensor, reg_max: int) -> Tensor:
    """xyxy box + anchor centre → (l, t, r, b) distances, clipped to [0, reg_max).

    All inputs must be in the **same unit** (grid-cell or pixel).
    """
    lt = anchor_pts - bboxes[..., :2]
    rb = bboxes[..., 2:] - anchor_pts
    return torch.cat([lt, rb], dim=-1).clamp(0, reg_max - 1 - 1e-3)


def bbox_iou(box1: Tensor, box2: Tensor, ciou: bool = False, eps: float = 1e-7) -> Tensor:
    """IoU (or CIoU) between matched pairs of xyxy boxes.

    CIoU adds a penalty for centre-distance and aspect-ratio difference,
    which guides the regressor to predict tighter, better-aligned boxes.

    Args:
        box1, box2: [N, 4] xyxy
        ciou:       compute Complete IoU

    Returns:
        [N] values in [0, 1]  (or slightly negative for CIoU penalty).
    """
    b1x1, b1y1, b1x2, b1y2 = box1.unbind(-1)
    b2x1, b2y1, b2x2, b2y2 = box2.unbind(-1)

    ix1 = torch.max(b1x1, b2x1)
    iy1 = torch.max(b1y1, b2y1)
    ix2 = torch.min(b1x2, b2x2)
    iy2 = torch.min(b1y2, b2y2)

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    w1, h1 = b1x2 - b1x1, b1y2 - b1y1
    w2, h2 = b2x2 - b2x1, b2y2 - b2y1
    union  = w1 * h1 + w2 * h2 - inter + eps
    iou    = inter / union

    if not ciou:
        return iou

    # Enclosing box diagonal squared
    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # Centre distance squared
    rho2 = ((b2x1 + b2x2 - b1x1 - b1x2) ** 2 +
            (b2y1 + b2y2 - b1y1 - b1y2) ** 2) / 4

    # Aspect-ratio consistency term v and its trade-off weight alpha
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    return iou - (rho2 / c2 + v * alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Task-Aligned Assigner
# ─────────────────────────────────────────────────────────────────────────────

class TaskAlignedAssigner(nn.Module):
    """Soft assignment of GT boxes to anchor points.

    Alignment metric per (anchor i, GT j):

        t_ij = s_ij ^ alpha  *  u_ij ^ beta

    where s_ij is the predicted probability for GT j's class at anchor i,
    and u_ij is the IoU between the predicted box and GT j.

    Top-k anchors per GT are selected; conflicts are resolved by keeping
    whichever GT has the higher alignment score for that anchor.
    """

    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0) -> None:
        super().__init__()
        self.topk  = topk
        self.alpha = alpha
        self.beta  = beta

    @torch.no_grad()
    def forward(
        self,
        pd_scores: Tensor,   # [B, N, nc]   sigmoid class scores
        pd_bboxes: Tensor,   # [B, N, 4]    decoded xyxy pixel coords
        anc_pts:   Tensor,   # [N, 2]       anchor centres
        gt_labels: Tensor,   # [B, M]       class indices (−1 = padding)
        gt_bboxes: Tensor,   # [B, M, 4]    xyxy pixel coords
        mask_gt:   Tensor,   # [B, M]       bool: True = real GT
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assign GTs to anchors.

        Returns
        ───────
        assigned_labels : [B, N]     class index (−1 for background)
        assigned_bboxes : [B, N, 4]  GT box for each foreground anchor
        assigned_scores : [B, N]     soft alignment score (used for cls target)
        fg_mask         : [B, N]     bool: True = foreground
        """
        B, M = gt_labels.shape
        N    = pd_scores.shape[1]

        # ── Pairwise IoU [B, N, M] ─────────────────────────────────────────
        p = pd_bboxes.unsqueeze(2)     # [B, N, 1, 4]
        g = gt_bboxes.unsqueeze(1)     # [B, 1, M, 4]
        ix1 = torch.max(p[..., 0], g[..., 0])
        iy1 = torch.max(p[..., 1], g[..., 1])
        ix2 = torch.min(p[..., 2], g[..., 2])
        iy2 = torch.min(p[..., 3], g[..., 3])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        ap = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
        ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
        iou = inter / (ap + ag - inter + 1e-7)          # [B, N, M]

        # ── Predicted score for each GT's class [B, N, M] ─────────────────
        gt_cls_idx = gt_labels[:, None, :].expand(B, N, M).clamp(0)
        pd_s_gt    = pd_scores.gather(-1, gt_cls_idx)   # [B, N, M]

        # ── Alignment metric ────────────────────────────────────────────────
        align = (pd_s_gt ** self.alpha) * (iou ** self.beta)
        align = align * mask_gt[:, None, :]             # zero-out padding GTs

        # ── Top-k selection ─────────────────────────────────────────────────
        topk     = min(self.topk, N)
        topk_val, _ = align.topk(topk, dim=1)           # [B, topk, M]
        thresh   = topk_val[:, -1:, :]                   # [B, 1, M]
        topk_mask = (align >= thresh) & mask_gt[:, None, :]

        # ── Resolve conflicts: keep highest-metric GT per anchor ─────────────
        align_sel      = align * topk_mask               # [B, N, M]
        best_score, best_gt_idx = align_sel.max(dim=-1)  # [B, N]
        fg_mask = best_score > 0

        # ── Gather assignments ───────────────────────────────────────────────
        best_gt_safe = best_gt_idx.clamp(0)
        assigned_labels = torch.where(
            fg_mask,
            gt_labels.gather(1, best_gt_safe),
            torch.full_like(best_gt_safe, -1),
        )
        idx4 = best_gt_safe.unsqueeze(-1).expand(B, N, 4)
        assigned_bboxes  = gt_bboxes.gather(1, idx4)

        # Normalise scores so they sum to ≤ 1 per anchor (used as BCE weight)
        assigned_scores = align_sel.amax(dim=-1) / (align_sel.sum(dim=-1) + 1e-9)

        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv11 loss
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv11Loss:
    """Compute box + cls + dfl loss from raw head outputs and target tensor.

    Args:
        model:    YOLOv11 instance (to read nc, reg_max, stride).
        box_gain: Weight for CIoU box regression loss.
        cls_gain: Weight for BCE classification loss.
        dfl_gain: Weight for distribution focal loss.
        topk:     Top-k per GT for the task-aligned assigner.
    """

    def __init__(
        self,
        model: nn.Module,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        topk: int = 10,
    ) -> None:
        det = model.detect
        self.nc       = det.nc
        self.reg_max  = det.reg_max
        self.stride   = det.stride
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.assigner = TaskAlignedAssigner(topk=topk)
        self.bce      = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(
        self,
        preds:   list[Tensor],   # training outputs: list of [B, 4·reg_max+nc, Hi, Wi]
        targets: Tensor,         # [K, 6]: [img_idx, cls, cx, cy, w, h]  normalised
    ) -> tuple[Tensor, dict[str, float]]:
        device = preds[0].device
        B      = preds[0].shape[0]

        loss_box = preds[0].new_zeros(1)
        loss_cls = preds[0].new_zeros(1)
        loss_dfl = preds[0].new_zeros(1)

        # ── Anchor grid ──────────────────────────────────────────────────────
        anc_pts, stride_t = make_anchors(preds, self.stride.to(device))  # [N,2], [N,1]
        N = anc_pts.shape[0]

        # ── Flatten predictions → [B, N, no] ────────────────────────────────
        flat = torch.cat(
            [p.flatten(2) for p in preds], dim=2
        ).transpose(1, 2)                                  # [B, N, no]
        pred_dist, pred_cls = flat.split([4 * self.reg_max, self.nc], dim=-1)

        # Decode DFL distribution → offsets (grid-cell units) → xyxy pixels
        dist4       = pred_dist.view(B, N, 4, self.reg_max).softmax(-1)
        bins        = torch.arange(self.reg_max, device=device, dtype=torch.float32)
        offsets_px  = (dist4 * bins).sum(-1) * stride_t.unsqueeze(0)    # [B, N, 4]
        pred_bboxes = dist2bbox(offsets_px, anc_pts.unsqueeze(0).expand(B, -1, -1))

        # ── GT tensors ───────────────────────────────────────────────────────
        input_size  = float(preds[0].shape[2]) * self.stride[0].item()
        gt_labels, gt_bboxes, mask_gt = self._build_gt_tensors(
            targets, B, input_size, device
        )

        # ── Task-aligned assignment ──────────────────────────────────────────
        assigned_labels, assigned_bboxes, assigned_scores, fg_mask = self.assigner(
            pred_cls.sigmoid().detach(),
            pred_bboxes.detach(),
            anc_pts,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        norm = fg_mask.sum().clamp(1).float()

        # ── Box loss (CIoU) ──────────────────────────────────────────────────
        if fg_mask.any():
            iou = bbox_iou(pred_bboxes[fg_mask], assigned_bboxes[fg_mask], ciou=True)
            loss_box = ((1.0 - iou) * assigned_scores[fg_mask]).sum() / norm

        # ── Classification loss (soft BCE) ───────────────────────────────────
        cls_target = torch.zeros_like(pred_cls)
        if fg_mask.any():
            fg_cls   = assigned_labels[fg_mask].clamp(0).long()
            fg_score = assigned_scores[fg_mask].unsqueeze(-1)
            cls_target[fg_mask].scatter_(-1, fg_cls.unsqueeze(-1), fg_score)
        loss_cls = self.bce(pred_cls, cls_target).sum() / norm

        # ── DFL loss ─────────────────────────────────────────────────────────
        if fg_mask.any():
            fg_anc = anc_pts.unsqueeze(0).expand(B, N, 2)[fg_mask]      # [K, 2] pixels
            fg_str = stride_t.unsqueeze(0).expand(B, N, 1)[fg_mask]     # [K, 1]
            fg_gt  = assigned_bboxes[fg_mask]                            # [K, 4] pixels
            # Convert to grid-cell distances for DFL target
            gt_dist = bbox2dist(fg_gt / fg_str, fg_anc / fg_str, self.reg_max)  # [K, 4]
            pd_dist = pred_dist[fg_mask].view(-1, 4, self.reg_max)              # [K, 4, R]
            loss_dfl = self._dfl_loss(pd_dist, gt_dist) / norm

        total = self.box_gain * loss_box + self.cls_gain * loss_cls + self.dfl_gain * loss_dfl
        return total, {
            "box": loss_box.item(),
            "cls": loss_cls.item(),
            "dfl": loss_dfl.item(),
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_gt_tensors(
        self,
        targets: Tensor,      # [K, 6]: img_idx, cls, cx, cy, w, h  (normalised)
        B: int,
        input_size: float,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pad GT annotations into [B, M, ...] tensors for batch processing.

        Returns:
            gt_labels: [B, M]     long, −1 for padding slots
            gt_bboxes: [B, M, 4]  xyxy pixel coords
            mask_gt:   [B, M]     bool
        """
        if targets.shape[0] == 0:
            return (
                torch.full((B, 1), -1, dtype=torch.long, device=device),
                torch.zeros(B, 1, 4, device=device),
                torch.zeros(B, 1, dtype=torch.bool, device=device),
            )

        M = max(int((targets[:, 0] == b).sum().item()) for b in range(B))
        M = max(M, 1)

        gt_labels = torch.full((B, M), -1, dtype=torch.long, device=device)
        gt_bboxes = torch.zeros(B, M, 4, device=device)
        mask_gt   = torch.zeros(B, M, dtype=torch.bool, device=device)

        for b in range(B):
            rows = targets[targets[:, 0] == b]    # [k, 6]
            k = rows.shape[0]
            if k == 0:
                continue
            cx, cy, w, h = (rows[:, i] * input_size for i in range(2, 6))
            gt_labels[b, :k] = rows[:, 1].long()
            gt_bboxes[b, :k] = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], -1)
            mask_gt[b, :k]   = True

        return gt_labels, gt_bboxes, mask_gt

    @staticmethod
    def _dfl_loss(pred_dist: Tensor, targets: Tensor) -> Tensor:
        """Soft cross-entropy loss for the DFL coordinate distribution.

        The target distance (e.g. 3.7) is split across the two adjacent bins
        (bin 3 gets weight 0.3, bin 4 gets weight 0.7), producing a smooth
        regression signal rather than a hard bin classification.

        Args:
            pred_dist: [K, 4, reg_max]  logits
            targets:   [K, 4]           float distances in [0, reg_max)

        Returns:
            scalar mean loss
        """
        K, _, R = pred_dist.shape
        tl = targets.long().clamp(0, R - 2)              # lower bin
        tr = tl + 1                                       # upper bin
        wl = (tr.float() - targets)                       # weight for lower
        wr = 1.0 - wl                                     # weight for upper

        logits = pred_dist.view(-1, R)
        loss   = (
            F.cross_entropy(logits, tl.view(-1), reduction="none") * wl.view(-1)
            + F.cross_entropy(logits, tr.view(-1), reduction="none") * wr.view(-1)
        )
        return loss.mean()
