import torch

from .base_assigner import BaseAssigner
from .assign_result import AssignResult
from ..geometry import bbox_overlaps


def calc_region(bbox, ratio, stride, featmap_size=None):
    # Base anchor locates in (stride - 1) * 0.5
    f_bbox = (bbox - (stride - 1) * 0.5) / stride
    x1 = torch.round((1 - ratio) * f_bbox[0] + ratio * f_bbox[2])
    y1 = torch.round((1 - ratio) * f_bbox[1] + ratio * f_bbox[3])
    x2 = torch.round(ratio * f_bbox[0] + (1 - ratio) * f_bbox[2])
    y2 = torch.round(ratio * f_bbox[1] + (1 - ratio) * f_bbox[3])
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1] - 1)
        y1 = y1.clamp(min=0, max=featmap_size[0] - 1)
        x2 = x2.clamp(min=0, max=featmap_size[1] - 1)
        y2 = y2.clamp(min=0, max=featmap_size[0] - 1)
    return (x1, y1, x2, y2)


def anchor_ctr_inside_region_flags(anchors, stride, region):
    x1, y1, x2, y2 = region
    f_anchors = (anchors - (stride - 1) * 0.5) / stride
    x = (f_anchors[:, 0] + f_anchors[:, 2]) * 0.5
    y = (f_anchors[:, 1] + f_anchors[:, 3]) * 0.5
    flags = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    return flags


def anchor_outside_flags(flat_anchors, valid_flags, img_shape,
                         allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    outside_flags = ~inside_flags
    return outside_flags


class MixIoURegionAnchorAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
    """
    # TODO add docs
    def __init__(self,
                 with_region=True,
                 with_iou=True,
                 center_ratio=0.2,
                 ignore_ratio=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.7,
                 min_pos_iou=0.3):
        assert with_region or with_iou
        self.with_region = with_region
        self.with_iou = with_iou
        self.center_ratio = center_ratio
        self.ignore_ratio = ignore_ratio
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou

    def assign(self, mlvl_anchors, mlvl_valid_flags, gt_bboxes, img_meta,
               featmap_sizes, anchor_scale, anchor_strides,
               gt_bboxes_ignore=None, gt_labels=None, allowed_border=0):
        """Assign gt to anchors.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. Assign every anchor to 0 (negative)
        For each gt_bboxes:
            2. Compute ignore flags based on ignore_region and iou_thr then
                assign -1 to anchors w.r.t. ignore flags
            3. Compute pos flags based on center_region and pos_iou_thr then
               assign gt_bboxes to anchors w.r.t. pos flags
            4. Compute ignore flags based on adjacent anchor lvl then
               assign -1 to anchors w.r.t. ignore flags
            5. Assign anchor outside of image to -1

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        # TODO support gt_bboxes_ignore
        if gt_bboxes.shape[0] == 0:
            raise ValueError('No gt bboxes')
        num_gts = gt_bboxes.shape[0]
        num_lvls = len(mlvl_anchors)
        r1 = (1 - self.center_ratio) / 2
        r2 = (1 - self.ignore_ratio) / 2

        scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) *
                           (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1))
        min_anchor_size = scale.new_full(
            (1, ), float(anchor_scale * anchor_strides[0]))
        target_lvls = torch.floor(
            torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
        target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()

        # 1. assign 0 (negative) by default
        mlvl_assigned_gt_inds = []
        mlvl_ignore_flags = []
        for lvl in range(num_lvls):
            h, w = featmap_sizes[lvl]
            assert h * w == mlvl_anchors[lvl].shape[0]
            assigned_gt_inds = gt_bboxes.new_full(
                (h * w, ), 0, dtype=torch.long)
            ignore_flags = torch.zeros_like(assigned_gt_inds)
            mlvl_assigned_gt_inds.append(assigned_gt_inds)
            mlvl_ignore_flags.append(ignore_flags)

        for gt_id in range(num_gts):
            lvl = target_lvls[gt_id].item()
            featmap_size = featmap_sizes[lvl]
            stride = anchor_strides[lvl]
            anchors = mlvl_anchors[lvl]
            gt_bbox = gt_bboxes[gt_id, :4]

            # Compute regions and iou overlaps
            if self.with_region:
                ignore_region = calc_region(gt_bbox, r2, stride, featmap_size)
                ctr_region = calc_region(gt_bbox, r1, stride, featmap_size)
            if self.with_iou:
                overlaps = bbox_overlaps(gt_bbox[None, :], anchors).squeeze(0)

            # 2. Assign -1 to ignore inds
            ignore_flags = 1
            if self.with_region:
                region_ignore_flags = anchor_ctr_inside_region_flags(
                    anchors, stride, ignore_region)
                ignore_flags = region_ignore_flags & ignore_flags
            if self.with_iou:
                # neg_iou_thr <= ignore_iou < pos_iou_thr
                iou_ignore_flags = overlaps < self.pos_iou_thr
                iou_ignore_flags &= (overlaps >= self.neg_iou_thr)
                ignore_flags = iou_ignore_flags & ignore_flags
            mlvl_assigned_gt_inds[lvl][ignore_flags > 0] = -1

            # 3. Assign gt_bboxes to pos inds
            # TODO: check potential bug that gt_id is not max overlap
            # due to override, and mismatch in pos_iou_thr
            pos_flags = 1
            if self.with_region:
                region_pos_flags = anchor_ctr_inside_region_flags(
                    anchors, stride, ctr_region)
                pos_flags = region_pos_flags & pos_flags
            if self.with_iou:
                iou_pos_flags = overlaps >= self.pos_iou_thr
                max_overlap = overlaps.max()
                if max_overlap >= self.min_pos_iou:
                    max_iou_flags = overlaps == max_overlap
                    iou_pos_flags |= max_iou_flags
                pos_flags = iou_pos_flags & pos_flags
            mlvl_assigned_gt_inds[lvl][pos_flags > 0] = gt_id + 1

            # 4. Assign -1 to ignore adjacent lvl
            if self.with_region:
                if lvl > 0:
                    d_lvl = lvl - 1
                    d_anchors = mlvl_anchors[d_lvl]
                    d_featmap_size = featmap_sizes[d_lvl]
                    d_stride = anchor_strides[d_lvl]
                    d_ignore_region = calc_region(
                        gt_bbox, d_stride, r2, d_featmap_size)
                    ignore_flags = anchor_ctr_inside_region_flags(
                        d_anchors, d_stride, d_ignore_region)
                    mlvl_ignore_flags[d_lvl][ignore_flags > 0] = 1
                if lvl < num_lvls - 1:
                    u_lvl = lvl + 1
                    u_anchors = mlvl_anchors[u_lvl]
                    u_featmap_size = featmap_sizes[u_lvl]
                    u_stride = anchor_strides[u_lvl]
                    u_ignore_region = calc_region(
                        gt_bbox, u_stride, r2, u_featmap_size)
                    ignore_flags = anchor_ctr_inside_region_flags(
                        u_anchors, u_stride, u_ignore_region)
                    mlvl_ignore_flags[u_lvl][ignore_flags > 0] = 1

        # 4. (cont.) Assign -1 to ignore adjacent lvl
        if self.with_region:
            for lvl in range(num_lvls):
                ignore_flags = mlvl_ignore_flags[lvl]
                mlvl_assigned_gt_inds[lvl][ignore_flags > 0] = -1

        # 5. Assign -1 to anchor outside of image
        flat_assigned_gt_inds = torch.cat(mlvl_assigned_gt_inds)
        flat_anchors = torch.cat(mlvl_anchors)
        flat_valid_flags = torch.cat(mlvl_valid_flags)
        assert (flat_assigned_gt_inds.shape[0] == flat_anchors.shape[0]
                == flat_valid_flags.shape[0])
        outside_flags = anchor_outside_flags(
            flat_anchors, flat_valid_flags,
            img_meta['img_shape'], allowed_border)
        flat_assigned_gt_inds[outside_flags] = -1

        if gt_labels is not None:
            assigned_labels = torch.zeros_like(flat_assigned_gt_inds)
            pos_inds = assigned_gt_inds > 0
            assigned_labels[pos_inds] = gt_labels[
                flat_assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, flat_assigned_gt_inds, None, labels=assigned_labels)
