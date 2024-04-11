#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
import cv2
def get_xywh(box):
    x_min=box[0]
    y_min=box[1]
    x_max= box[0]+box[2]
    y_max = box[1]+box[3]
    return x_min,y_min,x_max,y_max
def get_xywh_p(box):
    x_min=box[0]
    y_min=box[1]
    x_max= box[2]
    y_max = box[3]
    return x_min,y_min,x_max,y_max
def iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = get_xywh(bbox1)
    xmin2, ymin2, xmax2, ymax2 = get_xywh_p(bbox2[0])
    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    # 计算交集面积
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 ) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    # 计算交并比（交集/并集）
    iou = inter_area / (area1 + area2 - inter_area )  # 注意：这里inter_area不能乘以2，乘以2就相当于把交集部分挖空了
    return iou

def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input",  default='/home/skj/zw/VIS/F-R/SparseR-CNN-main/projects/KCDNet/output/inference/coco_instances_results.json',help="JSON file produced by the model")
    parser.add_argument("--output", default='/home/skj/zw/VIS/F-R/SparseR-CNN-main/projects/KCDNet/output/show',help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="M3FD_train")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)

    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)
    pre=[]
    label=[]

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        h,w,_=img.shape
        img_ir=img[:h//2,:w//2,:]
        img_vis = img[h // 2:, w // 2 :,:]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])


        vis = Visualizer(img_vis, metadata)

        vis_pred = vis.draw_instance_predictions(predictions).get_image()
        vis = Visualizer(img_vis, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()
        ir = Visualizer(img_ir, metadata)

        ir_pred = ir.draw_instance_predictions(predictions).get_image()

        ir = Visualizer(img_ir, metadata)
        ir_gt = ir.draw_dataset_dict(dic).get_image()




        cv2.imwrite(os.path.join(args.output+'/'+'vis', basename), vis_pred[:, :, ::-1])
        cv2.imwrite(os.path.join(str(args.output+'/'+'ir'), basename), ir_pred[:, :, ::-1])
        cv2.imwrite(os.path.join(str(args.output +'/'+ 'gt'), basename), vis_gt[:, :, ::-1])
