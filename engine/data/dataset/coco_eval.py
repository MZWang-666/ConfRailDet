"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import csv
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator',]


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def summarize_per_class(self, iou_type="bbox", output_file=None):
        if iou_type not in self.coco_eval:
            return []

        coco_eval = self.coco_eval[iou_type]
        if not hasattr(coco_eval, "eval") or not coco_eval.eval:
            return []

        precision = coco_eval.eval.get("precision")
        recall = coco_eval.eval.get("recall")
        if precision is None:
            return []

        params = coco_eval.params
        cat_ids = list(params.catIds)
        area_idx = _index_or_default(getattr(params, "areaRngLbl", []), "all", 0)
        max_det_idx = len(params.maxDets) - 1
        iou_thrs = np.array(params.iouThrs)
        iou50_idx = int(np.argmin(np.abs(iou_thrs - 0.50)))
        iou75_idx = int(np.argmin(np.abs(iou_thrs - 0.75)))

        rows = []
        for class_idx, cat_id in enumerate(cat_ids):
            class_precision = precision[:, :, class_idx, area_idx, max_det_idx]
            row = {
                "class_id": cat_id,
                "class_name": self._get_category_name(cat_id),
                "AP50": _mean_valid(class_precision[iou50_idx]),
                "AP75": _mean_valid(class_precision[iou75_idx]),
                "AP50:95": _mean_valid(class_precision),
            }

            if recall is not None:
                class_recall = recall[:, class_idx, area_idx, max_det_idx]
                row["AR50:95"] = _mean_valid(class_recall)

            rows.append(row)

        if not rows:
            return rows

        self._print_per_class_table(rows)

        if output_file is not None:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fieldnames = ["class_id", "class_name", "AP50", "AP75", "AP50:95"]
            if "AR50:95" in rows[0]:
                fieldnames.append("AR50:95")
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        key: _format_metric(row[key]) if isinstance(row[key], float) else row[key]
                        for key in fieldnames
                    })
            print(f"Per-class AP saved to: {output_file}")

        return rows

    def _get_category_name(self, cat_id):
        cat = self.coco_gt.cats.get(cat_id, {}) if hasattr(self.coco_gt, "cats") else {}
        return cat.get("name", str(cat_id))

    @staticmethod
    def _print_per_class_table(rows):
        has_ar = "AR50:95" in rows[0]
        if has_ar:
            print("\nPer-class bbox metrics:")
            print(f"{'class_name':<32} | {'AP50':>7} | {'AP75':>7} | {'AP50:95':>8} | {'AR50:95':>8}")
            print("-" * 75)
            for row in rows:
                print(
                    f"{row['class_name']:<32} | {_format_metric(row['AP50']):>7} | "
                    f"{_format_metric(row['AP75']):>7} | {_format_metric(row['AP50:95']):>8} | "
                    f"{_format_metric(row['AR50:95']):>8}"
                )
        else:
            print("\nPer-class bbox AP:")
            print(f"{'class_name':<32} | {'AP50':>7} | {'AP75':>7} | {'AP50:95':>8}")
            print("-" * 64)
            for row in rows:
                print(
                    f"{row['class_name']:<32} | {_format_metric(row['AP50']):>7} | "
                    f"{_format_metric(row['AP75']):>7} | {_format_metric(row['AP50:95']):>8}"
                )

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def _index_or_default(items, value, default):
    try:
        return list(items).index(value)
    except ValueError:
        return default


def _mean_valid(values):
    values = np.asarray(values)
    values = values[values > -1]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _format_metric(value):
    if not np.isfinite(value):
        return "nan"
    return f"{value * 100:.3f}"


def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
