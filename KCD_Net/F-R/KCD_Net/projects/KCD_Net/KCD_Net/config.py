# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_KCDNet_config(cfg):
    """
    Add config for KCDNet.
    """
    cfg.MODEL.KCDNet = CN()
    cfg.MODEL.KCDNet.NUM_CLASSES = 80
    cfg.MODEL.KCDNet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.KCDNet.NHEADS = 8
    cfg.MODEL.KCDNet.DROPOUT = 0.0
    cfg.MODEL.KCDNet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.KCDNet.ACTIVATION = 'relu'
    cfg.MODEL.KCDNet.HIDDEN_DIM = 256
    cfg.MODEL.KCDNet.NUM_CLS = 1
    cfg.MODEL.KCDNet.NUM_REG = 3
    cfg.MODEL.KCDNet.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.KCDNet.NUM_DYNAMIC = 2
    cfg.MODEL.KCDNet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.KCDNet.CLASS_WEIGHT = 2.0
    cfg.MODEL.KCDNet.GIOU_WEIGHT = 2.0
    cfg.MODEL.KCDNet.L1_WEIGHT = 5.0
    cfg.MODEL.KCDNet.DEEP_SUPERVISION = True
    cfg.MODEL.KCDNet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.KCDNet.USE_FOCAL = True
    cfg.MODEL.KCDNet.ALPHA = 0.25
    cfg.MODEL.KCDNet.GAMMA = 2.0
    cfg.MODEL.KCDNet.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
