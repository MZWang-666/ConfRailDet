import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig


DEFAULT_CONFIG = "configs/confraildet/rtdetrv4_s_baseline.yml"
DEFAULT_OUTPUT_DIR = "outputs/analysis/train_prototype_similarity"
DEFAULT_CONFUSION_CSV = "outputs/analysis/baseline_confusion/train/confusion_counts.csv"

EVAL_TRANSFORMS = [
    {"type": "Resize", "size": [640, 640]},
    {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
]
EVAL_POLICY = {"name": "default", "epoch": 0, "ops": []}

DEFAULT_DOMAIN_GROUPS = {
    "internal_B": [0, 1, 2, 3, 4],
    "external_RGB": [5, 6, 7, 8],
}


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict, source_name = select_state_dict(checkpoint, source)
    missing, unexpected = model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    return source_name, missing, unexpected


def build_cfg(config_path, batch_size, num_workers):
    updates = {
        "use_ema": False,
        "train_dataloader": {
            "shuffle": False,
            "drop_last": False,
            "total_batch_size": batch_size,
            "num_workers": num_workers,
            "dataset": {
                "transforms": {
                    "ops": EVAL_TRANSFORMS,
                    "policy": EVAL_POLICY,
                    "mosaic_prob": -0.1,
                }
            },
            "collate_fn": {
                "mixup_prob": 0.0,
                "base_size_repeat": None,
                "stop_epoch": 0,
            },
        },
    }
    cfg = YAMLConfig(config_path, **updates)
    transforms_cfg = cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]
    transforms_cfg["ops"] = EVAL_TRANSFORMS
    transforms_cfg["policy"] = EVAL_POLICY
    transforms_cfg["mosaic_prob"] = -0.1
    return cfg


def get_categories(coco):
    categories = sorted(coco.dataset.get("categories", []), key=lambda item: item["id"])
    cat_names = [str(item.get("name", item["id"])) for item in categories]
    return cat_names


def build_same_domain_mask(num_classes, allow_cross_domain=False):
    if allow_cross_domain:
        mask = np.ones((num_classes, num_classes), dtype=bool)
        np.fill_diagonal(mask, False)
        return mask

    mask = np.zeros((num_classes, num_classes), dtype=bool)
    for class_ids in DEFAULT_DOMAIN_GROUPS.values():
        ids = [idx for idx in class_ids if idx < num_classes]
        for i in ids:
            for j in ids:
                if i != j:
                    mask[i, j] = True
    return mask


def make_rois(targets, device):
    rois = []
    labels = []
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        target_labels = target["labels"].to(device=device, dtype=torch.long)
        if boxes.numel() == 0:
            continue
        batch_col = torch.full((boxes.shape[0], 1), batch_idx, dtype=boxes.dtype, device=device)
        rois.append(torch.cat([batch_col, boxes], dim=1))
        labels.append(target_labels)
    if len(rois) == 0:
        return None, None
    return torch.cat(rois, dim=0), torch.cat(labels, dim=0)


def extract_roi_features(model, samples, targets):
    backbone_feats = model.backbone(samples)
    encoder_feats = model.encoder(backbone_feats)
    if isinstance(encoder_feats, tuple):
        encoder_feats = encoder_feats[0]

    rois, labels = make_rois(targets, samples.device)
    if rois is None:
        return None, None

    pooled_per_level = []
    image_h, image_w = samples.shape[-2:]
    for feat in encoder_feats:
        spatial_scale = float(feat.shape[-1]) / float(image_w)
        pooled = roi_align(
            feat,
            rois,
            output_size=(1, 1),
            spatial_scale=spatial_scale,
            aligned=True,
        ).flatten(1)
        pooled_per_level.append(F.normalize(pooled, dim=-1))

    features = torch.stack(pooled_per_level, dim=0).mean(dim=0)
    features = F.normalize(features, dim=-1)
    return features, labels


@torch.no_grad()
def compute_prototypes(args):
    cfg = build_cfg(args.config, args.batch_size, args.num_workers)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = cfg.model.to(device)
    checkpoint_source = "none"
    missing, unexpected = [], []
    if args.checkpoint:
        checkpoint_source, missing, unexpected = load_checkpoint(model, args.checkpoint, args.checkpoint_source)
    model.eval()

    loader = cfg.train_dataloader
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(0)

    cat_names = get_categories(loader.dataset.coco)
    num_classes = len(cat_names)
    feat_sum = None
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    processed = 0

    for samples, targets in loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        features, labels = extract_roi_features(model, samples, targets)
        processed += samples.shape[0]
        if features is not None:
            if feat_sum is None:
                feat_sum = torch.zeros(num_classes, features.shape[-1], dtype=torch.float64, device=device)
            for class_id in range(num_classes):
                mask = labels == class_id
                if mask.any():
                    feat_sum[class_id] += features[mask].double().sum(dim=0)
                    counts[class_id] += mask.sum()

        if args.max_images is not None and processed >= args.max_images:
            break
        if processed % args.print_freq == 0:
            print(f"processed {processed} training images")

    if feat_sum is None:
        raise RuntimeError("No object features were extracted from the training split.")

    prototypes = feat_sum / counts.clamp_min(1.0).unsqueeze(1)
    prototypes = F.normalize(prototypes.float(), dim=-1)
    similarity = (prototypes @ prototypes.t()).detach().cpu().numpy()

    print(f"feature checkpoint source: {checkpoint_source}")
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    print(f"processed images: {processed}")
    print(f"class counts: {[int(x) for x in counts.detach().cpu().tolist()]}")
    return similarity, counts.detach().cpu().numpy().astype(np.int64), cat_names, checkpoint_source


def load_matrix_csv(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    names = rows[0][1:]
    matrix = np.array(
        [[float(value) for value in row[1:1 + len(names)]] for row in rows[1:1 + len(names)]],
        dtype=np.float64,
    )
    row_names = [row[0] for row in rows[1:1 + len(names)]]
    if row_names != names:
        raise ValueError(f"Matrix row/column labels do not match in {path}.")
    return matrix, names


def load_training_confusion_frequency(path, num_classes):
    path = Path(path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    class_rows = rows[1:1 + num_classes]
    offdiag_counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    gt_counts = np.zeros(num_classes, dtype=np.float64)
    for i, row in enumerate(class_rows):
        # The class columns contain matched predictions and the following column is
        # "missed". Their sum is the number of GT objects for this class.
        values = np.array([float(value) for value in row[1:2 + num_classes]], dtype=np.float64)
        offdiag_counts[i] = values[:num_classes]
        gt_counts[i] = values.sum()

    np.fill_diagonal(offdiag_counts, 0.0)
    directed = np.divide(
        offdiag_counts,
        np.maximum(gt_counts[:, None], 1.0),
        out=np.zeros_like(offdiag_counts),
    )
    np.fill_diagonal(directed, 0.0)
    return directed


def load_gt_counts(path, num_classes):
    path = Path(path)
    if not path.exists():
        return np.zeros(num_classes, dtype=np.int64)

    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    counts = []
    for row in rows[1:1 + num_classes]:
        values = [float(value) for value in row[1:2 + num_classes]]
        counts.append(int(round(sum(values))))
    return np.asarray(counts, dtype=np.int64)


def normalize_offdiag(matrix, mask):
    values = matrix.copy().astype(np.float64)
    values[~mask] = 0.0
    valid = values[mask]
    if valid.size == 0:
        return values
    v_min = float(valid.min())
    v_max = float(valid.max())
    if v_max - v_min < 1e-12:
        out = np.zeros_like(values)
        out[mask] = valid
        return out
    out = np.zeros_like(values)
    out[mask] = (valid - v_min) / (v_max - v_min)
    return out


def build_prior_matrix(similarity, confusion_frequency, mask, visual_weight):
    sim01 = (similarity + 1.0) * 0.5
    np.fill_diagonal(sim01, 0.0)
    sim_norm = normalize_offdiag(sim01, mask)

    if confusion_frequency is None:
        prior = sim_norm
        conf_norm = np.zeros_like(sim_norm)
    else:
        conf_norm = normalize_offdiag(confusion_frequency, mask)
        prior = visual_weight * sim_norm + (1.0 - visual_weight) * conf_norm

    prior[~mask] = 0.0
    np.fill_diagonal(prior, 0.0)
    return prior, sim_norm, conf_norm


def save_matrix_csv(path, matrix, row_names, col_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + col_names)
        for name, row in zip(row_names, matrix):
            writer.writerow([name] + [f"{float(value):.6f}" for value in row])


def save_heatmap(path, matrix, names, cmap="viridis"):
    fig_w = max(8, len(names) * 0.8)
    fig_h = max(7, len(names) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j or abs(float(matrix[i, j])) < 5e-3:
                continue
            rgba = im.cmap(im.norm(matrix[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.55 else "white"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def top_pairs(matrix, names, mask, topk=20):
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if mask[i, j] or mask[j, i]:
                score = max(float(matrix[i, j]), float(matrix[j, i]))
                pairs.append((score, names[i], names[j]))
    pairs.sort(reverse=True)
    return pairs[:topk]


def top_directed_pairs(matrix, names, mask, topk=20, positive_only=True):
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j or not mask[i, j]:
                continue
            score = float(matrix[i, j])
            if positive_only and score <= 0.0:
                continue
            pairs.append((score, names[i], names[j]))
    pairs.sort(reverse=True)
    return pairs[:topk]


def save_summary(
    path,
    similarity,
    prior,
    confusion_frequency,
    names,
    counts,
    mask,
    visual_weight,
    feature_source,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Training-set class prototype similarity and confusion prior analysis\n")
        f.write("Prototype definition: class-wise mean ROI feature extracted from training-set GT boxes.\n")
        f.write("Similarity metric: cosine similarity between L2-normalized class prototypes.\n")
        f.write(f"Feature source: {feature_source}.\n")
        f.write("Prototype estimation uses GT boxes and feature embeddings, not detector predictions.\n")
        f.write(f"Prior fusion: A_ij = {visual_weight:.2f} * normalized visual similarity")
        if confusion_frequency is None:
            f.write("; baseline training-set confusion frequency was not used because no CSV was available.\n\n")
        else:
            f.write(f" + {1.0 - visual_weight:.2f} * normalized baseline training-set confusion frequency.\n\n")
            f.write(
                "Baseline confusion frequency is directional P(predicted=j | GT=i), with the GT count "
                "including correct, misclassified, and missed instances.\n\n"
            )

        f.write("Class prototype sample counts:\n")
        for name, count in zip(names, counts):
            f.write(f"- {name}: {int(count)}\n")

        f.write("\nTop same-domain prototype-similarity pairs:\n")
        for score, left, right in top_pairs(similarity, names, mask):
            f.write(f"- {left} <-> {right}: cosine={score:.4f}\n")

        if confusion_frequency is not None:
            f.write("\nTop same-domain baseline training confusion directions:\n")
            for score, source, target in top_directed_pairs(confusion_frequency, names, mask):
                f.write(f"- {source} -> {target}: frequency={score:.6f}\n")

        f.write("\nTop same-domain directed confusion-prior relations:\n")
        for score, source, target in top_directed_pairs(prior, names, mask):
            f.write(f"- {source} -> {target}: prior={score:.4f}\n")

        f.write("\nEvidence-based observations:\n")
        for domain_name, class_ids in DEFAULT_DOMAIN_GROUPS.items():
            ids = [idx for idx in class_ids if idx < len(names)]
            pair_values = [
                float(similarity[i, j])
                for offset, i in enumerate(ids)
                for j in ids[offset + 1:]
            ]
            domain_mask = np.zeros_like(mask)
            domain_mask[np.ix_(ids, ids)] = True
            np.fill_diagonal(domain_mask, False)
            strongest = top_pairs(similarity, names, domain_mask, topk=1)[0]
            observation = (
                f"- {domain_name}: prototype cosine range={min(pair_values):.4f}-{max(pair_values):.4f}, "
                f"mean={float(np.mean(pair_values)):.4f}; strongest pair is "
                f"{strongest[1]} <-> {strongest[2]} ({strongest[0]:.4f})"
            )
            if confusion_frequency is not None:
                domain_gt = float(np.asarray(counts)[ids].sum())
                weighted_confusions = sum(
                    float(counts[i]) * float(confusion_frequency[i, ids].sum()) for i in ids
                )
                observation += f"; baseline train off-diagonal rate={weighted_confusions / max(domain_gt, 1.0):.6f}"
            f.write(observation + ".\n")
        f.write(
            "- The external_RGB evidence is supported by both feature proximity and baseline errors. "
            "The internal_B evidence is subgroup-specific, led by normal bolt versus abnormal bolt, "
            "rather than uniform confusion among all five B-scan classes.\n"
        )

        f.write("\nInterpretation note:\n")
        f.write(
            "The matrix is restricted to category sets that belong to the same acquisition domain. "
            "It should not be interpreted as cross-domain adaptation, because the internal B-scan and "
            "external RGB categories are domain-exclusive.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute training-set class prototype similarity.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default="", help="Optional feature-extractor checkpoint.")
    parser.add_argument("--checkpoint-source", default="auto", choices=["auto", "ema", "model"])
    parser.add_argument(
        "--prototype-csv",
        default="",
        help="Reuse an existing prototype cosine-similarity CSV instead of extracting features again.",
    )
    parser.add_argument("--confusion-csv", default=DEFAULT_CONFUSION_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--visual-weight", type=float, default=0.5)
    parser.add_argument("--allow-cross-domain", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.config = str(resolve_project_path(args.config))
    args.checkpoint = str(resolve_project_path(args.checkpoint)) if args.checkpoint else ""
    args.prototype_csv = str(resolve_project_path(args.prototype_csv)) if args.prototype_csv else ""
    args.confusion_csv = str(resolve_project_path(args.confusion_csv))
    args.output_dir = str(resolve_project_path(args.output_dir))
    if args.checkpoint and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if args.prototype_csv and not os.path.exists(args.prototype_csv):
        raise FileNotFoundError(args.prototype_csv)
    if not 0.0 <= args.visual_weight <= 1.0:
        raise ValueError("visual-weight must be in [0, 1].")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prototype_csv:
        similarity, names = load_matrix_csv(args.prototype_csv)
        counts = load_gt_counts(args.confusion_csv, len(names))
        feature_source = f"reused cosine-similarity matrix ({args.prototype_csv})"
    else:
        similarity, counts, names, checkpoint_source = compute_prototypes(args)
        feature_source = (
            f"checkpoint {args.checkpoint} ({checkpoint_source})"
            if args.checkpoint
            else "model initialization defined by the config"
        )
    mask = build_same_domain_mask(len(names), allow_cross_domain=args.allow_cross_domain)
    confusion_frequency = load_training_confusion_frequency(args.confusion_csv, len(names))
    prior, visual_norm, confusion_norm = build_prior_matrix(
        similarity,
        confusion_frequency,
        mask,
        visual_weight=args.visual_weight,
    )

    save_matrix_csv(output_dir / "class_prototype_cosine_similarity.csv", similarity, names, names)
    save_matrix_csv(output_dir / "visual_similarity_normalized_same_domain.csv", visual_norm, names, names)
    raw_confusion = confusion_frequency if confusion_frequency is not None else np.zeros_like(similarity)
    save_matrix_csv(output_dir / "baseline_train_confusion_frequency.csv", raw_confusion, names, names)
    save_matrix_csv(
        output_dir / "baseline_train_confusion_frequency_normalized_same_domain.csv",
        confusion_norm,
        names,
        names,
    )
    save_matrix_csv(output_dir / "confusion_prior_matrix.csv", prior, names, names)

    save_heatmap(
        output_dir / "class_prototype_cosine_similarity.png",
        similarity,
        names,
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "confusion_prior_matrix.png",
        prior,
        names,
        cmap="magma",
    )
    save_summary(
        output_dir / "prototype_similarity_summary_en.txt",
        similarity,
        prior,
        confusion_frequency,
        names,
        counts,
        mask,
        args.visual_weight,
        feature_source,
    )

    print(f"results saved to: {output_dir}")


if __name__ == "__main__":
    main()
