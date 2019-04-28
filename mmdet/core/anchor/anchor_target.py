import torch

from ..bbox import (assign_and_sample, build_assigner, PseudoSampler,
                    bbox2delta, bbox_overlaps)
from ..utils import multi_apply


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


def anchor_ctr_inside_region_inds(anchors, stride, region):
    flags = anchor_ctr_inside_region_flags(anchors, stride, region)
    return torch.nonzero(flags).squeeze(1)


def ca_anchor_target(anchor_list,
                     gt_bboxes_list,
                     featmap_sizes,
                     anchor_scale,
                     anchor_strides,
                     target_means,
                     target_stds,
                     cfg,
                     label_channels=1):
    """Cascade Anchoring: get anchor target, w.r.t. iou threshold
        and region ratio.
    1. Init all target and weight to be negative
    2. Compute anchors and gt_bboxes on featmap
    3. Compute center and ignore regions for each gt_bbox
    4. Assign target and weight for ignore region
    5. Compute ctr_pos_inds for center region
    6. Compute iou_pos_inds for pos iou thr
    7. Get intersection of ctr_pos_inds and iou_pos_inds then assign
        target and weight
    """
    # TODO support multi-class
    assert label_channels == 1, 'Support binary class label currently'
    num_imgs = len(gt_bboxes_list)
    num_lvls = len(featmap_sizes)
    r1 = (1 - cfg.center_ratio) / 2
    r2 = (1 - cfg.ignore_ratio) / 2
    all_label_targets = []
    all_label_weights = []
    all_bbox_targets = []
    all_bbox_weights = []
    all_ignore_map = []

    for lvl_id in range(num_lvls):
        h, w = featmap_sizes[lvl_id]
        label_targets = torch.zeros(
            num_imgs, h * w, 1, device='cuda', dtype=torch.float32)
        label_weights = torch.full_like(label_targets, -1)
        bbox_targets = torch.zeros(
            num_imgs, h * w, 4, device='cuda', dtype=torch.float32)
        bbox_weights = torch.zeros_like(bbox_targets)
        ignore_map = torch.zeros_like(label_targets)
        all_label_targets.append(label_targets)
        all_label_weights.append(label_weights)
        all_bbox_targets.append(bbox_targets)
        all_bbox_weights.append(bbox_weights)
        all_ignore_map.append(ignore_map)

    for img_id in range(num_imgs):
        gt_bboxes = gt_bboxes_list[img_id]
        scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) *
                           (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1))
        min_anchor_size = scale.new_full(
            (1, ), float(anchor_scale * anchor_strides[0]))
        target_lvls = torch.floor(
            torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
        target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()

        for gt_id in range(gt_bboxes.size(0)):
            lvl = target_lvls[gt_id].item()
            featmap_size = featmap_sizes[lvl]
            stride = anchor_strides[lvl]

            anchors = anchor_list[img_id][lvl]
            gt_bbox = gt_bboxes[gt_id, :4]

            # Compute ignore_inds and ctr_pos_inds
            ignore_region = calc_region(gt_bbox, r2, stride, featmap_size)
            ctr_region = calc_region(gt_bbox, r1, stride, featmap_size)
            ignore_inds = anchor_ctr_inside_region_inds(
                anchors, stride, ignore_region)
            ctr_pos_flags = anchor_ctr_inside_region_flags(
                anchors, stride, ctr_region)

            # Compute iou_pos_inds
            overlaps = bbox_overlaps(
                gt_bbox[None, :], anchors).squeeze(0)
            iou_pos_flags = overlaps > cfg.pos_iou_thr
            # Assign pos for highest IoU
            max_overlap = overlaps.max()
            if max_overlap > cfg.min_pos_iou:
                max_iou_flags = overlaps == max_overlap
            else:
                max_iou_flags = torch.zeros_like(iou_pos_flags).byte()
            iou_pos_flags = iou_pos_flags | max_iou_flags

            # Merge ctr_pos_flags and iou_pos_flags
            # TODO: test iou
            assert cfg.with_iou is False
            pos_flags = (ctr_pos_flags & iou_pos_flags if cfg.with_iou
                         else ctr_pos_flags)
            pos_inds = torch.nonzero(pos_flags).squeeze(1)

            all_label_targets[lvl][img_id, pos_inds, 0] = 1
            all_label_weights[lvl][img_id, ignore_inds, 0] = 0
            all_label_weights[lvl][img_id, pos_inds, 0] = 1

            num_pos = pos_inds.shape[0]
            gt_bbox_all_pos = gt_bbox.expand(num_pos, 4)
            bbox_targets = bbox2delta(
                anchors[pos_inds], gt_bbox_all_pos, target_means,
                target_stds)
            all_bbox_targets[lvl][img_id, pos_inds, :] = bbox_targets
            all_bbox_weights[lvl][img_id, pos_inds, :] = 1

            if lvl > 0:
                d_lvl = lvl - 1
                d_anchors = anchor_list[img_id][d_lvl]
                d_featmap_size = featmap_sizes[d_lvl]
                d_stride = anchor_strides[d_lvl]
                d_ignore_region = calc_region(
                    gt_bbox, d_stride, r2, d_featmap_size)
                ignore_inds = anchor_ctr_inside_region_inds(
                    d_anchors, d_stride, d_ignore_region)
                all_ignore_map[d_lvl][img_id, ignore_inds, 0] = 1

            if lvl < num_lvls - 1:
                u_lvl = lvl + 1
                u_anchors = anchor_list[img_id][u_lvl]
                u_featmap_size = featmap_sizes[u_lvl]
                u_stride = anchor_strides[u_lvl]
                u_ignore_region = calc_region(
                    gt_bbox, u_stride, r2, u_featmap_size)
                ignore_inds = anchor_ctr_inside_region_inds(
                    u_anchors, u_stride, u_ignore_region)
                all_ignore_map[u_lvl][img_id, ignore_inds, 0] = 1

    for lvl_id in range(num_lvls):
        all_label_weights[lvl_id][(all_label_weights[lvl_id] < 0)
                                  & (all_ignore_map[lvl_id] > 0)] = 0
        all_label_weights[lvl_id][all_label_weights[lvl_id] < 0] = 0.1
    num_total_samples = sum(
        [t.size(0) * t.size(1) for t in all_label_targets]) / 200
    return (all_label_targets, all_label_weights, all_bbox_targets,
            all_bbox_weights, num_total_samples)


def anchor_offset(anchor_list, anchor_strides, featmap_sizes):
    """ Get offest for deformable conv based on anchor shape
    NOTE: currently support deformable kernel_size=3 and dilation=1

    Args:
        anchor_list (list[list[tensor])): [NI, NLVL, NA, 4] list of
            multi-level anchors
            anchor_strides (list): anchor stride of each level
    Returns:
        offset_list (list[tensor]): [NLVL, NA, 2, 18]: offset of 3x3 deformable
        kernel.
    """
    def _shape_offset(anchors, stride):
        # currently support kernel_size=3 and dilation=1
        ks = 3
        dilation = 1
        pad = (ks - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        xx, yy = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        pad = (ks - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)  # return order matters
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        w = (anchors[:, 2] - anchors[:, 0] + 1) / stride
        h = (anchors[:, 3] - anchors[:, 1] + 1) / stride
        w = w / (ks - 1) - dilation
        h = h / (ks - 1) - dilation
        offset_x = w[:, None] * xx  # (NA, ks**2)
        offset_y = h[:, None] * yy  # (NA, ks**2)
        return offset_x, offset_y

    def _ctr_offset(anchors, stride, featmap_size):
        feat_h, feat_w = featmap_size
        assert len(anchors) == feat_h * feat_w

        x = (anchors[:, 0] + anchors[:, 2]) * 0.5
        y = (anchors[:, 1] + anchors[:, 3]) * 0.5
        # compute centers on feature map
        x = (x - (stride - 1) * 0.5) / stride
        y = (y - (stride - 1) * 0.5) / stride
        # compute predefine centers
        xx = torch.arange(0, feat_w, device=anchors.device)
        yy = torch.arange(0, feat_h, device=anchors.device)
        yy, xx = torch.meshgrid(yy, xx)
        xx = xx.reshape(-1).type_as(x)
        yy = yy.reshape(-1).type_as(y)

        offset_x = x - xx  # (NA, )
        offset_y = y - yy  # (NA, )
        return offset_x, offset_y

    num_imgs = len(anchor_list)
    num_lvls = len(anchor_list[0])
    dtype = anchor_list[0][0].dtype
    device = anchor_list[0][0].device
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

    offset_list = []
    for i in range(num_imgs):
        mlvl_offset = []
        for lvl in range(num_lvls):
            c_offset_x, c_offset_y = _ctr_offset(
                anchor_list[i][lvl], anchor_strides[lvl], featmap_sizes[lvl])
            s_offset_x, s_offset_y = _shape_offset(
                anchor_list[i][lvl], anchor_strides[lvl])

            # offset = ctr_offset + shape_offset
            offset_x = s_offset_x + c_offset_x[:, None]
            offset_y = s_offset_y + c_offset_y[:, None]

            # offset order (y0, x0, y1, x0, .., y9, x8, y9, x9)
            offset = torch.stack([offset_y, offset_x], dim=-1)
            offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
            mlvl_offset.append(offset)
        offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
    offset_list = images_to_levels(offset_list, num_level_anchors)
    return offset_list


def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    _anchor_list = []
    _valid_flag_list = []
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        _anchor_list.append(torch.cat(anchor_list[i]))
        _valid_flag_list.append(torch.cat(valid_flag_list[i]))

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         anchor_target_single,
         _anchor_list,
         _valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    rois_list = images_to_levels(_anchor_list, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, rois_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
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
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
