from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox,
                        multi_apply, weighted_cross_entropy, weighted_smoothl1,
                        weighted_binary_cross_entropy,
                        weighted_sigmoid_focal_loss, multiclass_nms,
                        iou_loss, giou_loss, region_anchor_target)
from ..registry import HEADS


@HEADS.register_module
class CascadeAnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        use_sigmoid_cls (bool): Whether to use sigmoid loss for classification.
            (softmax by default)
        use_focal_loss (bool): Whether to use focal loss for classification.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 use_sigmoid_cls=False,
                 use_focal_loss=False,
                 with_cls=True,
                 sampling=True):
        super(CascadeAnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = use_sigmoid_cls
        self.use_focal_loss = use_focal_loss
        self.with_cls = with_cls
        self.sampling = sampling
        if use_focal_loss:
            assert not sampling

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes

    def _init_layers(self):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def forward_single(self, x, offset):
        raise NotImplementedError

    def forward(self, feats, offset_list=None):
        if offset_list is None:
            offset_list = [None for _ in range(len(feats))]
        return multi_apply(self.forward_single, feats, offset_list)

    def init_anchors(self, featmap_sizes, img_metas):
        """Init anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, rois, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg,
                    loss_weight):
        # classification loss
        if self.with_cls:
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                if self.use_focal_loss:
                    cls_criterion = weighted_sigmoid_focal_loss
                else:
                    cls_criterion = weighted_binary_cross_entropy
            else:
                if self.use_focal_loss:
                    raise NotImplementedError
                else:
                    cls_criterion = weighted_cross_entropy
            if self.use_focal_loss:
                loss_cls = cls_criterion(
                    cls_score,
                    labels,
                    label_weights,
                    gamma=cfg.gamma,
                    alpha=cfg.alpha,
                    avg_factor=num_total_samples)
            else:
                loss_cls = cls_criterion(cls_score, labels, label_weights,
                                         avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_loss_cfg = cfg.get('bbox_loss', None)
        if bbox_loss_cfg is None:
            loss_reg = weighted_smoothl1(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                beta=cfg.smoothl1_beta,
                avg_factor=num_total_samples)
        elif bbox_loss_cfg.type == 'IoU':
            rois = rois.reshape(-1, 4)
            loss_reg = iou_loss(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                rois,
                self.target_means,
                self.target_stds,
                reg_ratio=bbox_loss_cfg.reg_ratio,
                avg_factor=num_total_samples)
        elif bbox_loss_cfg.type == 'GIoU':
            rois = rois.reshape(-1, 4)
            loss_reg = giou_loss(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                rois,
                self.target_means,
                self.target_stds,
                reg_ratio=bbox_loss_cfg.reg_ratio,
                avg_factor=num_total_samples)
        else:
            raise Exception('Unknown config {}'.format(bbox_loss_cfg.type))
        if self.with_cls:
            return loss_cls * loss_weight, loss_reg * loss_weight
        return None, loss_reg * loss_weight

    def loss(self,
             anchor_list,
             valid_flag_list,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             loss_weight=1,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        assigner_type = cfg.assigner['type']
        if assigner_type == 'RegionAssigner':
            cls_reg_targets = region_anchor_target(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                featmap_sizes,
                self.anchor_scales[0],
                self.anchor_strides,
                self.target_means,
                self.target_stds,
                cfg,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                label_channels=label_channels,
                sampling=self.sampling)
        else:
            cls_reg_targets = anchor_target(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                self.target_means,
                self.target_stds,
                cfg,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                label_channels=label_channels,
                sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, rois_list, num_total_pos, num_total_neg
         ) = cls_reg_targets
        # num_total_samples = (num_total_pos + num_total_neg if self.sampling
        #                     else num_total_pos)
        if self.sampling:
            num_total_samples = num_total_pos + num_total_neg
        else:
            num_total_samples = sum([label.numel() for
                                     label in labels_list]) / 200.0
        losses = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            rois_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg,
            loss_weight=loss_weight)
        if self.with_cls:
            return dict(loss_cls=losses[0], loss_reg=losses[1])
        return dict(loss_reg=losses[1])

    def get_bboxes(self, anchor_list, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               anchor_list[img_id], img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def refine_bboxes(self, anchor_list, bbox_preds, img_metas):
        num_levels = len(bbox_preds)
        new_anchor_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                img_shape = img_metas[img_id]['img_shape']
                bboxes = delta2bbox(
                    anchor_list[img_id][i], bbox_pred, self.target_means,
                    self.target_stds, img_shape)
                mlvl_anchors.append(bboxes)
            new_anchor_list.append(mlvl_anchors)
        return new_anchor_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels
