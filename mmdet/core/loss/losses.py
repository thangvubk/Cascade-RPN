# TODO merge naive and weighted loss.
import torch
import torch.nn.functional as F

from ...ops import sigmoid_focal_loss


from mmdet.core import bbox_overlaps, delta2bbox


def weighted_nll_loss(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.nll_loss(pred, label, reduction='none')
    return torch.sum(raw * weight)[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor


def py_sigmoid_focal_loss(pred,
                          target,
                          weight,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return torch.sum(
        sigmoid_focal_loss(pred, target, gamma, alpha, 'none') * weight.view(
            -1, 1))[None] / avg_factor


def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def iou_loss(pred,
             target,
             weight,
             rois,
             target_means,
             target_stds,
             avg_factor=None,
             reg_ratio=1.):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    pos_inds = weight.nonzero().view(-1, 8)[:, 0]
    pos_mask = weight > 0
    pos_bbox_pred = pred[pos_mask].view(-1, 4)
    pos_bbox_target = target[pos_mask].view(-1, 4)
    pos_rois = rois[pos_inds, :] if rois.shape[1] == 4 else rois[pos_inds, 1:]
    pos_pred_bboxes = delta2bbox(pos_rois, pos_bbox_pred, target_means,
                                 target_stds)
    pos_target_bboxes = delta2bbox(pos_rois, pos_bbox_target, target_means,
                                   target_stds)
    ious = bbox_overlaps(pos_pred_bboxes, pos_target_bboxes, is_aligned=True)
    iou_distances = 1 - ious
    return torch.sum(iou_distances)[None] / avg_factor * reg_ratio


def giou_loss(pred,
              target,
              weight,
              rois,
              target_means,
              target_stds,
              avg_factor=None,
              reg_ratio=10.):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    pos_inds = weight.nonzero().view(-1, 8)[:, 0]
    pos_mask = weight > 0
    pos_bbox_pred = pred[pos_mask].view(-1, 4)
    pos_bbox_target = target[pos_mask].view(-1, 4)
    pos_rois = rois[pos_inds, :] if rois.shape[1] == 4 else rois[pos_inds, 1:]
    pos_pred_bboxes = delta2bbox(pos_rois, pos_bbox_pred, target_means,
                                 target_stds)
    pos_target_bboxes = delta2bbox(pos_rois, pos_bbox_target, target_means,
                                   target_stds)

    bboxes1 = pos_pred_bboxes
    bboxes2 = pos_target_bboxes
    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    enclose_x1y1 = torch.min(pos_pred_bboxes[:, :2], pos_target_bboxes[:, :2])
    enclose_x2y2 = torch.max(pos_pred_bboxes[:, 2:], pos_target_bboxes[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    u = ap + ag - overlap
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious
    return torch.sum(iou_distances)[None] / avg_factor * reg_ratio
