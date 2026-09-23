import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig


DEFAULT_CONFIG = "configs/confraildet/rtdetrv4_s_baseline.yml"
DEFAULT_CHECKPOINT = "checkpoints/rtdetrv4_s_baseline.pth"
DEFAULT_OUTPUT_DIR = "outputs/analysis/baseline_confusion"

EVAL_TRANSFORMS = [
    {"type": "Resize", "size": [640, 640]},
    {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
]
EVAL_POLICY = {"name": "default", "epoch": 0, "ops": []}


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def strip_module_prefix(state_dict):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def select_state_dict(checkpoint, source):
    if source == "ema" or (source == "auto" and "ema" in checkpoint):
        ema = checkpoint.get("ema", {})
        if isinstance(ema, dict) and "module" in ema:
            return ema["module"], "ema.module"

    if source == "model" or (source == "auto" and "model" in checkpoint):
        return checkpoint["model"], "model"

    if all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, "raw"

    raise KeyError("Could not find a model state dict in the checkpoint.")


def load_checkpoint(model, checkpoint_path, source="auto"):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict, source_name = select_state_dict(checkpoint, source)
    missing, unexpected = model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    return source_name, missing, unexpected


def build_cfg(
    config_path,
    split,
    batch_size,
    num_workers,
    img_folder=None,
    ann_file=None,
):
    updates = {
        "HGNetv2": {"pretrained": False},
        "use_ema": False,
    }

    if split == "train":
        updates["train_dataloader"] = {
            "shuffle": False,
            "drop_last": False,
            "total_batch_size": batch_size,
            "num_workers": num_workers,
            "dataset": {
                "transforms": {
                    "ops": EVAL_TRANSFORMS,
                    "policy": EVAL_POLICY,
                }
            },
            "collate_fn": {
                "mixup_prob": 0.0,
                "base_size_repeat": None,
                "stop_epoch": 0,
            },
        }
    else:
        updates["val_dataloader"] = {
            "total_batch_size": batch_size,
            "num_workers": num_workers,
        }
        dataset_updates = {}
        if img_folder is not None:
            dataset_updates["img_folder"] = str(img_folder)
        if ann_file is not None:
            dataset_updates["ann_file"] = str(ann_file)
        if dataset_updates:
            updates["val_dataloader"]["dataset"] = dataset_updates

    cfg = YAMLConfig(config_path, **updates)
    if split == "train":
        transforms_cfg = cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]
        transforms_cfg["ops"] = EVAL_TRANSFORMS
        transforms_cfg["policy"] = EVAL_POLICY
        transforms_cfg["mosaic_prob"] = -0.1
    return cfg


def get_loader(cfg, split):
    loader = cfg.train_dataloader if split == "train" else cfg.val_dataloader
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(0)
    return loader


def build_gt_by_image(coco):
    gt_by_image = defaultdict(list)
    for ann in coco.dataset.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        gt_by_image[int(ann["image_id"])].append({
            "category_id": int(ann["category_id"]),
            "box": [float(x), float(y), float(x + w), float(y + h)],
        })
    return gt_by_image


def get_categories(coco):
    categories = sorted(coco.dataset.get("categories", []), key=lambda item: item["id"])
    cat_ids = [int(item["id"]) for item in categories]
    cat_names = [str(item.get("name", item["id"])) for item in categories]
    return cat_ids, cat_names


def update_confusion_for_image(
    matrix,
    gt_items,
    prediction,
    cat_id_to_idx,
    iou_threshold,
    score_threshold,
):
    num_classes = len(cat_id_to_idx)
    if len(gt_items) == 0 and len(prediction["labels"]) == 0:
        return

    pred_scores = prediction["scores"].detach().cpu()
    keep = pred_scores >= score_threshold
    pred_boxes = prediction["boxes"].detach().cpu()[keep]
    pred_labels = prediction["labels"].detach().cpu()[keep]
    pred_scores = pred_scores[keep]

    gt_boxes = torch.tensor([item["box"] for item in gt_items], dtype=torch.float32)
    gt_labels = [int(item["category_id"]) for item in gt_items]

    used_gt = set()
    if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
        ious = box_iou(pred_boxes, gt_boxes)
    else:
        ious = torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=torch.float32)

    pred_order = torch.argsort(pred_scores, descending=True).tolist()
    for pred_idx in pred_order:
        pred_cat = int(pred_labels[pred_idx].item())
        if pred_cat not in cat_id_to_idx:
            continue
        pred_col = cat_id_to_idx[pred_cat]

        best_gt = -1
        best_iou = -1.0
        for gt_idx in range(len(gt_items)):
            if gt_idx in used_gt:
                continue
            cur_iou = float(ious[pred_idx, gt_idx].item()) if ious.numel() > 0 else 0.0
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt = gt_idx

        if best_gt >= 0 and best_iou >= iou_threshold:
            gt_cat = gt_labels[best_gt]
            if gt_cat in cat_id_to_idx:
                matrix[cat_id_to_idx[gt_cat], pred_col] += 1
            used_gt.add(best_gt)
        else:
            matrix[num_classes, pred_col] += 1

    missed_col = num_classes
    for gt_idx, gt_cat in enumerate(gt_labels):
        if gt_idx not in used_gt and gt_cat in cat_id_to_idx:
            matrix[cat_id_to_idx[gt_cat], missed_col] += 1


def save_matrix_csv(path, matrix, row_names, col_names, normalized=False):
    values = matrix.astype(np.float64)
    if normalized:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, np.maximum(row_sums, 1.0))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + col_names)
        for row_name, row in zip(row_names, values):
            if normalized:
                writer.writerow([row_name] + [f"{value:.6f}" for value in row])
            else:
                writer.writerow([row_name] + [int(value) for value in row])


def save_heatmap(path, matrix, row_names, col_names, normalized=True):
    values = matrix.astype(np.float64)
    if normalized:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, np.maximum(row_sums, 1.0))

    fig_w = max(9, len(col_names) * 0.75)
    fig_h = max(7, len(row_names) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=max(1e-9, values.max()))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            raw_value = int(matrix[i, j])
            if raw_value == 0:
                continue
            text = f"{values[i, j]:.2f}" if normalized else str(raw_value)
            rgba = im.cmap(im.norm(values[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.55 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(path, matrix, cat_names, iou_threshold, score_threshold, split, model_name):
    num_classes = len(cat_names)
    class_matrix = matrix[:num_classes, :num_classes]
    missed = matrix[:num_classes, num_classes]
    false_positive = matrix[num_classes, :num_classes]
    gt_count = class_matrix.sum(axis=1) + missed
    pred_count = class_matrix.sum(axis=0) + false_positive

    correct = np.diag(class_matrix)
    recall = np.divide(correct, np.maximum(gt_count, 1))
    precision = np.divide(correct, np.maximum(pred_count, 1))

    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and class_matrix[i, j] > 0:
                pairs.append((int(class_matrix[i, j]), cat_names[i], cat_names[j]))
    pairs.sort(reverse=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{model_name} confusion analysis on the {split} split\n")
        f.write(f"IoU threshold: {iou_threshold:.2f}\n")
        f.write(f"Score threshold: {score_threshold:.2f}\n\n")

        f.write("Per-class recall and precision:\n")
        for name, rec, pre, gt, pred, miss in zip(cat_names, recall, precision, gt_count, pred_count, missed):
            f.write(
                f"- {name}: recall={rec:.4f}, precision={pre:.4f}, "
                f"gt={int(gt)}, predicted={int(pred)}, missed={int(miss)}\n"
            )

        f.write("\nTop off-diagonal confusion pairs:\n")
        if pairs:
            for count, gt_name, pred_name in pairs[:20]:
                f.write(f"- {gt_name} -> {pred_name}: {count}\n")
        else:
            f.write("- No off-diagonal class confusions were observed under the selected thresholds.\n")

        f.write("\nFalse positives by predicted class:\n")
        for name, count in sorted(zip(cat_names, false_positive), key=lambda item: int(item[1]), reverse=True):
            f.write(f"- {name}: {int(count)}\n")


@torch.no_grad()
def evaluate_split(args, split):
    cfg = build_cfg(
        args.config,
        split,
        args.batch_size,
        args.num_workers,
        img_folder=args.img_folder if split == "test" else None,
        ann_file=args.ann_file if split == "test" else None,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = cfg.model.to(device)
    postprocessor = cfg.postprocessor.to(device)
    source_name, missing, unexpected = load_checkpoint(model, args.checkpoint, args.checkpoint_source)
    model.eval()
    postprocessor.eval()

    loader = get_loader(cfg, split)
    dataset = loader.dataset
    coco = dataset.coco
    gt_by_image = build_gt_by_image(coco)
    cat_ids, cat_names = get_categories(coco)
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    num_classes = len(cat_ids)
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    processed = 0

    for samples, targets in loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        outputs = model(samples)
        orig_target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
        predictions = postprocessor(outputs, orig_target_sizes)

        for target, prediction in zip(targets, predictions):
            image_id = int(target["image_id"].item())
            update_confusion_for_image(
                matrix,
                gt_by_image.get(image_id, []),
                prediction,
                cat_id_to_idx,
                args.iou_threshold,
                args.score_threshold,
            )
            processed += 1
            if args.max_images is not None and processed >= args.max_images:
                break

        if args.max_images is not None and processed >= args.max_images:
            break

        if processed % args.print_freq == 0:
            print(f"[{split}] processed {processed} images")

    split_dir = Path(args.output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    row_names = cat_names + ["false_positive"]
    col_names = cat_names + ["missed"]

    save_matrix_csv(split_dir / "confusion_counts.csv", matrix, row_names, col_names, normalized=False)
    save_matrix_csv(split_dir / "confusion_row_normalized.csv", matrix, row_names, col_names, normalized=True)
    save_heatmap(
        split_dir / "confusion_row_normalized.png",
        matrix,
        row_names,
        col_names,
        normalized=True,
    )
    write_summary(
        split_dir / "confusion_summary_en.txt",
        matrix,
        cat_names,
        args.iou_threshold,
        args.score_threshold,
        split,
        args.model_name,
    )

    print(f"[{split}] checkpoint source: {source_name}")
    print(f"[{split}] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    print(f"[{split}] processed images: {processed}")
    print(f"[{split}] results saved to: {split_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute detection confusion statistics.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-source", default="auto", choices=["auto", "ema", "model"])
    parser.add_argument(
        "--split", default="both", choices=["train", "val", "test", "both"]
    )
    parser.add_argument(
        "--img-folder",
        default=None,
        help="Image directory used when --split=test.",
    )
    parser.add_argument(
        "--ann-file",
        default=None,
        help="COCO annotation file used when --split=test.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-name",
        default="RT-DETRv4-S baseline",
        help="Model name written to the English summary.",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--print-freq", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    args.config = str(resolve_project_path(args.config))
    args.checkpoint = str(resolve_project_path(args.checkpoint))
    args.output_dir = str(resolve_project_path(args.output_dir))

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if args.split == "test" and (args.img_folder is None or args.ann_file is None):
        raise ValueError("--img-folder and --ann-file are required for --split=test")
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        evaluate_split(args, split)


if __name__ == "__main__":
    main()
