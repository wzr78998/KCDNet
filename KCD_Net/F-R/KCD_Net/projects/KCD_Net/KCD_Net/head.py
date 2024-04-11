#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
KCDNet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        
        # Build heads.
        num_classes = cfg.MODEL.KCDNet.NUM_CLASSES
        d_model = cfg.MODEL.KCDNet.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.KCDNet.DIM_FEEDFORWARD
        nhead = cfg.MODEL.KCDNet.NHEADS
        dropout = cfg.MODEL.KCDNet.DROPOUT
        activation = cfg.MODEL.KCDNet.ACTIVATION
        num_heads = cfg.MODEL.KCDNet.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)        
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.return_intermediate = cfg.MODEL.KCDNet.DEEP_SUPERVISION
        
        # Init parameters.
        self.use_focal = cfg.MODEL.KCDNet.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.KCDNet.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features_vis, features_ir,init_bboxes, init_features):

        inter_class_logits = []
        inter_pred_bboxes = []

        bs_vis = len(features_vis[0])
        bs_ir = len(features_vis[0])
        bboxes = init_bboxes
        
        init_features = init_features[None].repeat(1, bs_vis, 1)
        proposal_features = init_features.clone()
        lab=0
        
        for rcnn_head in self.head_series:
            if lab!=5:
                class_logits, pred_bboxes, proposal_features= rcnn_head(features_vis, features_ir, bboxes,
                                                                                    proposal_features, self.box_pooler,
                                                                                    lab)
            else:
                class_logits, pred_bboxes, proposal_features,loss_data = rcnn_head(features_vis, features_ir,bboxes, proposal_features, self.box_pooler,lab)
            lab=lab+1

            if self.return_intermediate and lab!=6:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            if self.return_intermediate and lab==6:
                inter_class_logits.append(class_logits[2])
                inter_pred_bboxes.append(pred_bboxes[2])
            if lab!=6:
                bboxes = pred_bboxes.detach()
            if lab==6:
                bboxes = pred_bboxes[2].detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes),loss_data

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = cfg.MODEL.KCDNet.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)
        self.fusion=nn.Conv2d(256*2,256,1,1)
        self.graph= nn.Conv2d(256 * 2, 256, 1, 1)
        self.c_a = nn.Conv2d(256 * 2, 256, 1, 1)
        self.sp_att = nn.Conv2d(256 * 2, 1, 1, 1)

        # reg.
        num_reg = cfg.MODEL.KCDNet.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = cfg.MODEL.KCDNet.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4*2)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights


    def forward(self, features_vis, features_ir,bboxes, pro_features, pooler,lab):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)

        """
        if lab!=5:
            N, nr_boxes = bboxes.shape[:2]

            # roi_feature.
            proposal_boxes = list()
            for b in range(N):
                proposal_boxes.append(Boxes(bboxes[b]))
            roi_features_vis = pooler(features_vis, proposal_boxes)
            roi_features_vis = roi_features_vis.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
            roi_features_ir = pooler(features_ir, proposal_boxes)
            roi_features_ir = roi_features_ir.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
            roi_features_fusion = self.fusion(
                torch.cat([roi_features_vis, roi_features_ir], 2).permute(2, 0, 1)).permute(1, 2, 0)
            class_logits_fusion, pred_bboxes_fusion, obj_features_fusion ,var= self.pred_det(pro_features, N, nr_boxes,
                                                                                         roi_features_fusion, bboxes)
            return class_logits_fusion.view(N, nr_boxes, -1),pred_bboxes_fusion.view(N, nr_boxes, -1),obj_features_fusion
        else:
            N, nr_boxes = bboxes.shape[:2]

            # roi_feature.
            proposal_boxes = list()
            for b in range(N):
                proposal_boxes.append(Boxes(bboxes[b]))
            roi_features_vis = pooler(features_vis, proposal_boxes)
            roi_features_vis = roi_features_vis.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
            roi_features_ir = pooler(features_ir, proposal_boxes)
            roi_features_ir = roi_features_ir.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
            roi_features_fusion = self.fusion(
                torch.cat([roi_features_vis, roi_features_ir], 2).permute(2, 0, 1)).permute(1, 2, 0)
            class_logits_ir, pred_bboxes_ir, obj_features_ir,var_ir = self.pred_det(pro_features, N, nr_boxes, nn.Tanh()(roi_features_ir),
                                                                             bboxes)
            condition_logits_ir=self.get_condition_logits(class_logits_ir)

            class_logits_vis, pred_bboxes_vis, obj_features_vis,var_vis = self.pred_det(pro_features, N, nr_boxes,
                                                                                nn.Tanh()(roi_features_vis), bboxes)
            condition_logits_vis = self.get_condition_logits(class_logits_vis)
            roi_features_ir_b=roi_features_ir[:,condition_logits_ir<condition_logits_vis,:]
            logits_ir_b=class_logits_ir[condition_logits_ir<condition_logits_vis]
            roi_features_ir_t = roi_features_ir[:,condition_logits_ir>condition_logits_vis,:]
            logits_ir_t = class_logits_ir[condition_logits_ir > condition_logits_vis]
            roi_features_vis_b = roi_features_vis[:, condition_logits_vis < condition_logits_ir, :]
            logits_vis_b = class_logits_vis[condition_logits_vis < condition_logits_ir]

            roi_features_vis_t = roi_features_vis[:, condition_logits_vis > condition_logits_ir, :]
            logits_vis_t = class_logits_vis[condition_logits_vis > condition_logits_ir]
            roi_features_ir_t=self.c_b(logits_ir_t ,logits_vis_b,roi_features_ir_t , roi_features_vis_b)
            roi_features_vis_t = self.c_b(logits_vis_t, logits_ir_b, roi_features_vis_t, roi_features_ir_b)
            roi_features_ir[:,condition_logits_ir > condition_logits_vis,:]=roi_features_ir_t
            roi_features_vis[:, condition_logits_vis> condition_logits_ir, :] = roi_features_vis_t

            var_vis=torch.mean(var_vis,1)
            var_ir = torch.mean(var_ir, 1)
            A1=1/nn.Sigmoid()(torch.mean(condition_logits_vis))
            A2= 1/nn.Sigmoid()(torch.mean(condition_logits_ir))
            A3 = 1/nn.Sigmoid()(torch.mean(var_vis))
            A4=1/nn.Sigmoid()(torch.mean(var_ir))

            if A1/A2>1:
                B1=A1/A2-1
                B2=0
            else:
                B1=0
                B2=1-A1/A2
            if A3/A4>1:
                B3=A3/A4-1
                B4=0
            else:
                B3=0
                B4=1-A3/A4



            roi_features_ir_b1 = roi_features_ir[:, var_vis > var_ir, :]
            logits_ir_b1 = class_logits_ir[var_vis >var_ir]

            roi_features_ir_t1 = roi_features_ir[:, var_vis < var_ir, :]
            logits_ir_t1 = class_logits_ir[var_vis < var_ir]
            roi_features_vis_b1 = roi_features_vis[:, var_vis < var_ir, :]
            logits_vis_b1 = class_logits_vis[var_vis <var_ir]

            roi_features_vis_t1 = roi_features_vis[:, var_vis > var_ir, :]
            logits_vis_t1 = class_logits_vis[var_vis > var_ir]
            roi_features_ir_t1 = self.s_b(logits_ir_t1 ,logits_vis_b1,roi_features_ir_t1 , roi_features_vis_b1)
            roi_features_vis_t1 = self.s_b(logits_vis_t1, logits_ir_b1, roi_features_vis_t1,roi_features_ir_b1)
            roi_features_ir[:, var_vis > var_ir, :] = roi_features_vis_t1
            roi_features_vis[:, var_vis < var_ir, :] = roi_features_ir_t1
            class_logits_fusion, pred_bboxes_fusion, obj_features_fusion,var_fusion = self.pred_det(pro_features, N, nr_boxes,
                                                                                         roi_features_fusion, bboxes)
            class_logits = [class_logits_ir.view(N, nr_boxes, -1), class_logits_vis.view(N, nr_boxes, -1),
                            class_logits_fusion.view(N, nr_boxes, -1)]
            pred_bboxes = [pred_bboxes_ir.view(N, nr_boxes, -1), pred_bboxes_vis.view(N, nr_boxes, -1),
                           pred_bboxes_fusion.view(N, nr_boxes, -1)]
            obj_features = [obj_features_ir, obj_features_vis, obj_features_fusion]

            return class_logits, pred_bboxes, obj_features,[pred_bboxes_vis,  class_logits_vis,pred_bboxes_ir,  class_logits_ir,B1,B2,B3,B4]



    

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes
    def pred_det(self,pro_features,N,nr_boxes,roi_features,bboxes):
        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes,
                                                                                             self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)

        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas0 = self.bboxes_delta(reg_feature)
        bboxes_deltas=bboxes_deltas0[:,:4]+bboxes_deltas0[:,4:]*torch.rand_like(bboxes_deltas0[:,4:])
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        return class_logits,pred_bboxes,obj_features,bboxes_deltas0[:,4:]
    def get_condition_logits(self,class_logits):
        class_logits_soft = nn.Softmax(1)(class_logits)
        ind=torch.argmax(class_logits_soft, 1)

        for i in range(ind.shape[0]):
            class_logits_soft[i,0]=class_logits_soft[i,ind[i]]


        condi_logits = class_logits_soft[:,0]*torch.log(class_logits_soft[:,0])
        return condi_logits

    def mean_roi(self,features, labels):
        """
        计算具有相同伪标签的同一模态ROI特征的均值ROI。

        参数:
        features - 维度为 (49, 200, 256) 的ROI特征张量。
        labels - 200个ROI特征的伪标签张量。

        返回:
        一个字典，键为伪标签，值为对应标签的均值ROI特征张量。
        """
        unique_labels = torch.unique(labels)
        mean_rois = {}
        label_ind=[]

        for label in unique_labels:
            # 找到具有当前伪标签的所有ROI特征
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            label_ind.append(label_indices)
            selected_features = features[:, label_indices, :]

            # 计算均值
            mean_feature = torch.mean(selected_features, dim=1)

            mean_rois[label.item()] = mean_feature



        return mean_rois,label_ind
    def c_b(self,class_logits_t,class_logits_b,roi_features_vis,roi_features_ir):
        c_know, label_ind = self.mean_roi(roi_features_vis, torch.argmax(class_logits_t, 1))
        c_know1, label_ind1 = self.mean_roi(roi_features_ir, torch.argmax(class_logits_b, 1))
        ii = 0
        for (key, var), (key1, var1) in zip(c_know.items(), c_know1.items()):
            c_k = var.unsqueeze(1).repeat(1, len(label_ind[ii]), 1)
            c_k= nn.Tanh()(
                self.graph(torch.cat([roi_features_vis[:, label_ind[ii], :], c_k], 2).permute(2, 0, 1)).permute(1, 2,
                                                                                                                0)) + roi_features_vis[
                                                                                                                      :,
                                                                                                                      label_ind[
                                                                                                                          ii],
                                                                                                                      :]

            roi_features_vis[:, label_ind[ii], :]=nn.Tanh()(self.c_a(torch.cat([roi_features_vis[:, label_ind[ii], :], c_k], 2).permute(2, 0, 1)).permute(1, 2,
                                                                                                                0)) + roi_features_vis[
                                                                                                                      :,
                                                                                                                      label_ind[
                                                                                                                          ii],
                                                                                                                      :]


            # c_k = var1.unsqueeze(1).repeat(1, len(label_ind1[ii]), 1)
            # roi_features_ir[:, label_ind1[ii], :] = nn.Tanh()(
            #     self.graph(torch.cat([roi_features_ir[:, label_ind1[ii], :], c_k], 2).permute(2, 0, 1)).permute(1, 2,
            #                                                                                                     0)) + roi_features_ir[
            #                                                                                                           :,
            #                                                                                                           label_ind1[
            #                                                                                                               ii],
            #                                                                                                           :]
            # roi_features_ir_att = self.sp_att(torch.cat(
            #     [roi_features_vis[:, label_ind[ii], :], c_know[key].unsqueeze(1).repeat(1, len(label_ind[ii]), 1)],
            #     2).permute(2, 0, 1)).permute(1, 2, 0)
            # roi_features_ir[:, label_ind[ii], :] = roi_features_ir[:, label_ind[ii], :] * roi_features_ir_att
            ii = ii + 1
        return roi_features_vis
    def s_b(self,class_logits_t,class_logits_b,roi_features_vis,roi_features_ir):
        c_know, label_ind = self.mean_roi(roi_features_vis, torch.argmax(class_logits_t, 1))
        c_know1, label_ind1 = self.mean_roi(roi_features_ir, torch.argmax(class_logits_b, 1))
        ii = 0
        for (key, var), (key1, var1) in zip(c_know.items(), c_know1.items()):


            roi_features_vis_att = self.sp_att(torch.cat(
                [roi_features_ir[:, label_ind1[ii], :], c_know1[key1].unsqueeze(1).repeat(1, len(label_ind1[ii]), 1)],
                2).permute(2, 0, 1)).permute(1, 2, 0)
            roi_features_vis[:, label_ind1[ii], :] = roi_features_vis[:, label_ind1[ii], :] * roi_features_vis_att
            ii = ii + 1



        return roi_features_vis


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.KCDNet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.KCDNet.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.KCDNet.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
