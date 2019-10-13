
# Cascade RPN

We provide the code for reproducing experiment results of Cascade RPN

- NeurIPS 2019 spotlight paper: [pdf](https://arxiv.org/abs/1909.06720).
- This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection) framework.
- If this work is helpful for your research, please cite Cascade RPN and mmdetection

```
@inproceedings{vu2019cascade,
  title={Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution},
  author={Vu, Thang and Jang, Hyunjun and Pham, Trung X and Yoo, Chang D},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2019}
}

@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal = {arXiv preprint arXiv:1906.07155},
  year    = {2019}
}
```

## Benchmark
### Region proposal performance
| Method | Backbone | Style | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR 1000 |                Download                |
|:------:|:--------:|:-----:|:--------:|:-------------------:|:--------------:|:-------:|:--------------------------------------:|
|   RPN  | R-50-FPN | caffe |     -    |          -          |        -       |   58.3  |                  model                 |
|  CRPN  | R-50-FPN | caffe |     -    |          -          |        -       |   71.7  | [model](http://bit.ly/cascade_rpn_r50) |

### Detection performance
|     Method    |   Proposal  | Backbone |  Style  | Schedule | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                   Download                   |
|:-------------:|:-----------:|:--------:|:-------:|:--------:|:--------:|:-------------------:|:--------------:|:------:|:--------------------------------------------:|
|   Fast R-CNN  |     RPN     | R-50-FPN |  caffe  |    1x    |    3.5   |        0.250        |      16.5      |  36.9  |                       -                      |
|   Fast R-CNN  | Cascade RPN | R-50-FPN |  caffe  |    1x    |    3.5   |        0.250        |      16.5      |  40.0  |     [model](http://bit.ly/crpn_fast_r50)     |
|  Faster R-CNN |     RPN     | R-50-FPN |  caffe  |    1x    |    3.8   |        0.353        |      13.6      |  37.0  |                       -                      |
|  Faster R-CNN | Cascade RPN | R-50-FPN |  caffe  |    1x    |    4.6   |        0.561        |      11.1      |  40.5  |    [model](http://bit.ly/crpn_faster_r50)    |
| Cascade R-CNN |     RPN     | R-50-FPN | pytorch |    1x    |    4.1   |        0.455        |      11.9      |  40.8  |                       -                      |
| Cascade R-CNN | Cascade RPN | R-50-FPN | pytorch |    1x    |    5.2   |        0.650        |       9.6      |  41.6  | [model](http://bit.ly/crpn_cascade_rcnn_r50) |

## Setup
Please follow official [installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md) and [getting_started](https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md) guides.

##  Testing
``./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]``

Note:
- Config files of Cascade RPN and accompanied detectors are located in [cascade_rpn](https://github.com/thangvubk/Cascade-RPN/tree/master/configs/cascade_rpn)
- Eval metrics are ``proposal_fast`` and ``bbox`` for region proposal and detection, respectively

Example of cascade rpn eval on 8 gpus:

```
./tools/dish_test.sh configs/cascade_rpn/cascade_rpn_r50_fpn_1x.py \
    checkpoint/cascade_rpn_r50_fpn_1x_20191008-d3e01c91.pth 8 --out \
    results/results.pkl --eval proposal_fast
```

## Training
``./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [--validate] [other_optional_args]``

Note: We train Cascade RPN and accompanied detectors with 8 GPUs and 2 img/GPU. If your configuration is different, please follow the [Linear Scaling Rule](https://github.com/thangvubk/Cascade-RPN/blob/master/docs/GETTING_STARTED.md#train-a-model).

## TODO
- [x] Release Cascade RPN code base
- [x] Release baseline models
- [ ] Release more models
