from torch import nn
import mmcv

from mmdet.core import tensor2imgs, anchor_offset
from .base import BaseDetector
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS


@DETECTORS.register_module
class CascadeRPN(BaseDetector, RPNTestMixin):

    def __init__(self,
                 num_stages,
                 backbone,
                 neck,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(CascadeRPN, self).__init__()
        assert num_stages == len(rpn_head)
        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck) if neck is not None else None

        self.rpn_head = nn.ModuleList()
        for head in rpn_head:
            self.rpn_head.append(builder.build_head(rpn_head))

        self.rpn_head = builder.build_head(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(CascadeRPN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            self.neck.init_weights()
        for i in range(self.num_stages):
            self.rpn_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_meta, gt_bboxes=None):
        x = self.extract_feat(img)
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        anchor_list, valid_flag_list = self.rpn_head[0].init_anchors(
            featmap_sizes, img_meta)
        losses = dict()

        for i in range(self.num_stages):
            rpn_train_cfg = self.train_cfg.rpn[i]
            rpn_head = self.rpn_head[i]

            offset_list = anchor_offset(
                anchor_list, rpn_head.anchor_strides, featmap_sizes)
            cls_score, bbox_pred = rpn_head(x, offset_list)
            rpn_loss_inputs = (
                anchor_list, valid_flag_list, cls_score, bbox_pred,
                gt_bboxes, img_meta, rpn_train_cfg)
            stage_loss = rpn_head.loss(*rpn_loss_inputs)
            for name, value in stage_loss.items():
                losses['s{}.{}'.format(i, name)] = value

            # refine boxes
            if i < self.num_stages - 1:
                anchor_list = rpn_head.refine_bboxes(
                    anchor_list, bbox_pred, img_meta)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        anchor_list, _ = self.rpn_head[0].init_anchors(featmap_sizes, img_meta)
        # TODO test ms_score
        # ms_score = [0 for lvl in range(len(x))]

        for i in range(self.num_stages):
            rpn_head = self.rpn_head[i]
            offset_list = anchor_offset(
                anchor_list, rpn_head.anchor_strides, featmap_sizes)
            cls_score, bbox_pred = rpn_head(x, offset_list)
            # ms_score = [ms_score[lvl] + cls_score[lvl] / self.num_stages
            #             for lvl in range(len(cls_score))]
            if i < self.num_stages - 1:
                anchor_list = rpn_head.refine_bboxes(
                    anchor_list, bbox_pred, img_meta)

        proposal_list = self.rpn_head[-1].get_bboxes(
            anchor_list, cls_score, bbox_pred, img_meta, self.test_cfg.rpn)

        if rescale:
            for proposals, meta in zip(proposal_list, img_meta):
                proposals[:, :4] /= meta['scale_factor']
        # TODO: remove this restriction
        return proposal_list[0].cpu().numpy()

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def show_result(self, data, result, img_norm_cfg, dataset=None, top_k=20):
        """Show RPN proposals on the image.

        Although we assume batch size is 1, this method supports arbitrary
        batch size.
        """
        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            mmcv.imshow_bboxes(img_show, result, top_k=top_k)
