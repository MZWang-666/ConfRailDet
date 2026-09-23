"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .confusion_head import build_confusion_relation
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized, is_main_process
from ..core import register

import logging

_logger = logging.getLogger(__name__)

DIW_DEFAULT_CLASS_NAMES = [
    'rail end',
    'damage to rail head',
    'normal bolt',
    'abnormal bolt',
    'bonding hole',
    'Spalling',
    'Wheel Burn',
    'Squat',
    'Corrugation',
]

DIW_DEFAULT_DOMAIN_NAMES = {
    0: 'external_RGB',
    1: 'internal_B',
}

DIW_DEFAULT_GROUPS = {
    'head': (1, 2, 4),
    'medium': (0, 5, 7),
    'tail': (3, 6, 8),
}


@register()
class RTv4Criterion(nn.Module):
    """ This class computes the loss for RT-DETRv4.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 num_classes=80,
                 reg_max=32,
                 boxes_weight_format=None,
                 share_matched_indices=False,
                 mal_alpha=None,
                 use_uni_set=True,
                 distill_adaptive_params=None,
                 use_diw_loss=False,
                 class_counts=None,
                 domain_class_counts=None,
                 diw_version='v1',
                 diw_alpha=0.5,
                 diw_beta=0.5,
                 diw_gamma=0.25,
                 diw_min=0.5,
                 diw_max=5.0,
                 diw_alpha_global=None,
                 diw_beta_intra_domain=None,
                 diw_gamma_image=None,
                 diw_weight_min=None,
                 diw_weight_max=None,
                 diw_normalize_weight=False,
                 diw_apply_to_main_cls=True,
                 diw_apply_to_aux_cls=True,
                 diw_apply_to_encoder_loss=False,
                 diw_apply_to_matcher_cost=False,
                 diw_log_weight_stats=True,
                 diw_log_first_n_batches=5,
                 confusion_domain_class_ids=None,
                 confusion_margin=0.2,
                 confusion_topk=2,
                 confusion_prior_mode='none',
                 confusion_prior_path=None,
                 confusion_prior_strength=1.0,
                 ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set

        self.distill_adaptive_params = distill_adaptive_params
        self.use_diw_loss = self._to_bool(use_diw_loss)
        self.diw_version = str(diw_version or 'v1').lower()
        if self.diw_version not in ('v1', 'v2'):
            raise ValueError(f"Unsupported diw_version: {diw_version}. Expected 'v1' or 'v2'.")
        self.diw_alpha = float(diw_alpha)
        self.diw_beta = float(diw_beta)
        self.diw_gamma = float(diw_gamma)
        self.diw_min = float(diw_min)
        self.diw_max = float(diw_max)
        if self.diw_min > self.diw_max:
            raise ValueError("diw_min should be no larger than diw_max.")

        self.diw_alpha_global = float(diw_alpha_global) if diw_alpha_global is not None else self.diw_alpha
        self.diw_beta_intra_domain = (
            float(diw_beta_intra_domain) if diw_beta_intra_domain is not None else self.diw_beta
        )
        self.diw_gamma_image = float(diw_gamma_image) if diw_gamma_image is not None else self.diw_gamma
        self.diw_weight_min = float(diw_weight_min) if diw_weight_min is not None else self.diw_min
        self.diw_weight_max = float(diw_weight_max) if diw_weight_max is not None else self.diw_max
        if self.diw_weight_min > self.diw_weight_max:
            raise ValueError("diw_weight_min should be no larger than diw_weight_max.")

        self.diw_normalize_weight = self._to_bool(diw_normalize_weight)
        self.diw_apply_to_main_cls = self._to_bool(diw_apply_to_main_cls)
        self.diw_apply_to_aux_cls = self._to_bool(diw_apply_to_aux_cls)
        self.diw_apply_to_encoder_loss = self._to_bool(diw_apply_to_encoder_loss)
        self.diw_apply_to_matcher_cost = self._to_bool(diw_apply_to_matcher_cost)
        self.diw_log_weight_stats = self._to_bool(diw_log_weight_stats)
        self.diw_log_first_n_batches = int(diw_log_first_n_batches)
        self._diw_image_log_count = 0
        self.confusion_margin = float(confusion_margin)
        self.confusion_topk = int(confusion_topk)
        if self.confusion_topk < 1:
            raise ValueError("confusion_topk must be >= 1.")
        self.confusion_prior_mode = str(confusion_prior_mode or 'none').lower()
        if self.confusion_prior_mode not in ('none', 'margin'):
            raise ValueError("confusion_prior_mode must be 'none' or 'margin'.")
        self.confusion_prior_strength = float(confusion_prior_strength)
        if self.confusion_prior_strength < 0:
            raise ValueError("confusion_prior_strength must be non-negative.")
        self.confusion_prior_path = str(confusion_prior_path or '')

        global_counts = self._as_class_count_tensor(class_counts, num_classes)
        domain_counts = self._as_domain_count_tensor(domain_class_counts, num_classes)
        global_weight = self._counts_to_inverse_weights(global_counts) if global_counts is not None else None
        domain_weight = self._counts_to_inverse_weights(domain_counts) if domain_counts is not None else None
        intra_class_weight, class_to_domain = self._build_intra_class_weight(domain_counts, domain_weight, num_classes)
        static_class_weight, static_norm_factor = self._build_static_class_weight(
            global_weight, intra_class_weight, num_classes)
        self.register_buffer('diw_global_weight', global_weight, persistent=False)
        self.register_buffer('diw_domain_weight', domain_weight, persistent=False)
        self.register_buffer(
            'diw_domain_count_mask',
            (domain_counts > 0) if domain_counts is not None else None,
            persistent=False,
        )
        self.register_buffer('diw_intra_class_weight', intra_class_weight, persistent=False)
        self.register_buffer('diw_class_to_domain', class_to_domain, persistent=False)
        self.register_buffer('diw_static_class_weight', static_class_weight, persistent=False)
        self.register_buffer('diw_static_norm_factor', static_norm_factor, persistent=False)
        self.diw_has_stats = global_weight is not None or domain_weight is not None

        self._configure_matcher_diw_cost()
        self._log_diw_static_weight_summary()
        confusion_relation = build_confusion_relation(num_classes, confusion_domain_class_ids)
        self.register_buffer('confusion_relation', confusion_relation, persistent=False)

        confusion_prior = torch.zeros(num_classes, num_classes, dtype=torch.float32)
        if self.confusion_prior_mode == 'margin':
            confusion_prior = self._load_confusion_prior(self.confusion_prior_path, num_classes)
            confusion_prior = confusion_prior * confusion_relation.to(dtype=confusion_prior.dtype)
            _logger.info(
                "Loaded confusion prior for margin modulation from %s (strength=%.3f).",
                self.confusion_prior_path,
                self.confusion_prior_strength,
            )
        self.register_buffer('confusion_prior', confusion_prior, persistent=False)


    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        return bool(value)

    @staticmethod
    def _load_confusion_prior(path, num_classes):
        if not path:
            raise ValueError("confusion_prior_path is required when confusion_prior_mode='margin'.")

        prior_path = Path(path).expanduser()
        if not prior_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            root_relative = project_root / prior_path
            prior_path = root_relative if root_relative.exists() else Path.cwd() / prior_path
        if not prior_path.exists():
            raise FileNotFoundError(f"Confusion prior matrix not found: {prior_path}")

        with prior_path.open('r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.reader(f))
        if len(rows) < num_classes + 1 or len(rows[0]) < num_classes + 1:
            raise ValueError(
                f"Confusion prior must contain a {num_classes}x{num_classes} matrix: {prior_path}")

        values = []
        for row_idx, row in enumerate(rows[1:num_classes + 1]):
            if len(row) < num_classes + 1:
                raise ValueError(f"Confusion prior row {row_idx} is too short: {prior_path}")
            try:
                values.append([float(value) for value in row[1:num_classes + 1]])
            except ValueError as exc:
                raise ValueError(
                    f"Confusion prior contains a non-numeric value in row {row_idx}: {prior_path}") from exc

        prior = torch.tensor(values, dtype=torch.float32)
        if not torch.isfinite(prior).all():
            raise ValueError(f"Confusion prior contains NaN or Inf: {prior_path}")
        if prior.min().item() < 0.0 or prior.max().item() > 1.0 + 1e-6:
            raise ValueError(f"Confusion prior values must be in [0, 1]: {prior_path}")
        prior = prior.clamp(0.0, 1.0)
        prior.fill_diagonal_(0.0)
        return prior

    @staticmethod
    def _as_class_count_tensor(class_counts, num_classes):
        if class_counts is None:
            return None

        if isinstance(class_counts, dict):
            counts = torch.zeros(num_classes, dtype=torch.float32)
            for cls_id, count in class_counts.items():
                cls_id = int(cls_id)
                if 0 <= cls_id < num_classes:
                    counts[cls_id] = float(count)
        elif torch.is_tensor(class_counts):
            counts = class_counts.detach().float().flatten().cpu()
        else:
            counts = torch.as_tensor(class_counts, dtype=torch.float32).flatten()

        if counts.numel() == 0:
            return None
        if counts.numel() != num_classes:
            _logger.warning(
                "Ignore DIW class_counts because length %s does not match num_classes %s.",
                counts.numel(), num_classes)
            return None
        return counts

    @classmethod
    def _as_domain_count_tensor(cls, domain_class_counts, num_classes):
        if domain_class_counts is None:
            return None

        if isinstance(domain_class_counts, dict):
            domain_ids = [int(domain_id) for domain_id in domain_class_counts.keys()]
            if len(domain_ids) == 0:
                return None
            counts = torch.zeros(max(domain_ids) + 1, num_classes, dtype=torch.float32)
            for domain_id, row_counts in domain_class_counts.items():
                row = cls._as_class_count_tensor(row_counts, num_classes)
                if row is not None:
                    counts[int(domain_id)] = row
        elif torch.is_tensor(domain_class_counts):
            counts = domain_class_counts.detach().float().cpu()
        else:
            counts = torch.as_tensor(domain_class_counts, dtype=torch.float32)

        if counts.numel() == 0:
            return None
        if counts.dim() == 1:
            if counts.numel() != num_classes:
                _logger.warning(
                    "Ignore DIW domain_class_counts because flat length %s does not match num_classes %s.",
                    counts.numel(), num_classes)
                return None
            counts = counts.unsqueeze(0)
        if counts.dim() != 2 or counts.shape[1] != num_classes:
            _logger.warning(
                "Ignore DIW domain_class_counts because shape %s is not [num_domains, %s].",
                tuple(counts.shape), num_classes)
            return None
        return counts

    @staticmethod
    def _counts_to_inverse_weights(counts):
        weights = torch.ones_like(counts, dtype=torch.float32)
        if counts.dim() == 1:
            valid = counts > 0
            if valid.any():
                weights[valid] = counts[valid].mean() / counts[valid]
        elif counts.dim() == 2:
            for domain_id in range(counts.shape[0]):
                valid = counts[domain_id] > 0
                if valid.any():
                    weights[domain_id, valid] = counts[domain_id, valid].mean() / counts[domain_id, valid]
        else:
            raise ValueError("DIW counts should be a 1D or 2D tensor.")
        return weights

    @staticmethod
    def _build_intra_class_weight(domain_counts, domain_weight, num_classes):
        if domain_counts is None or domain_weight is None:
            return None, None

        intra_weight = torch.ones(num_classes, dtype=torch.float32)
        class_to_domain = torch.full((num_classes,), -1, dtype=torch.long)
        for cls_id in range(num_classes):
            valid_domains = torch.nonzero(domain_counts[:, cls_id] > 0, as_tuple=False).flatten()
            if valid_domains.numel() == 0:
                continue
            domain_id = int(valid_domains[0].item())
            class_to_domain[cls_id] = domain_id
            intra_weight[cls_id] = domain_weight[domain_id, cls_id]
        return intra_weight, class_to_domain

    def _get_diw_exponents(self):
        if self.diw_version == 'v2':
            return self.diw_alpha_global, self.diw_beta_intra_domain, self.diw_gamma_image
        return self.diw_alpha, self.diw_beta, self.diw_gamma

    def _get_diw_clip_range(self):
        if self.diw_version == 'v2':
            return self.diw_weight_min, self.diw_weight_max
        return self.diw_min, self.diw_max

    def _build_static_class_weight(self, global_weight, intra_class_weight, num_classes):
        if global_weight is None and intra_class_weight is None:
            return None, None

        alpha, beta, _ = self._get_diw_exponents()
        weight = torch.ones(num_classes, dtype=torch.float32)
        if global_weight is not None:
            weight = weight * global_weight.float().pow(alpha)
        if intra_class_weight is not None:
            weight = weight * intra_class_weight.float().pow(beta)

        weight_min, weight_max = self._get_diw_clip_range()
        weight = weight.clamp(min=weight_min, max=weight_max)
        norm_factor = torch.ones((), dtype=torch.float32)
        if self.diw_version == 'v2' and self.diw_normalize_weight and weight.numel() > 0:
            norm_factor = weight.mean().clamp_min(1e-12)
            weight = (weight / norm_factor).clamp(min=weight_min, max=weight_max)
        return weight, norm_factor

    def _diw_log(self, message):
        _logger.info(message)
        if is_main_process():
            print(message)

    def _format_diw_values(self, values, precision=4):
        return '[' + ', '.join(f'{float(v):.{precision}f}' for v in values) + ']'

    def _get_diw_class_names(self):
        if self.num_classes == len(DIW_DEFAULT_CLASS_NAMES):
            return DIW_DEFAULT_CLASS_NAMES
        return [f'class_{idx}' for idx in range(self.num_classes)]

    def _log_diw_static_weight_summary(self):
        if not self.use_diw_loss or not self.diw_log_weight_stats:
            return
        if not self.diw_has_stats:
            self._diw_log('[DIW] use_diw_loss=True but class/domain statistics are missing; DIW is inactive.')
            return

        device = self.diw_static_class_weight.device if self.diw_static_class_weight is not None else torch.device('cpu')
        global_weight = self.diw_global_weight if self.diw_global_weight is not None else torch.ones(
            self.num_classes, dtype=torch.float32, device=device)
        intra_weight = self.diw_intra_class_weight if self.diw_intra_class_weight is not None else torch.ones(
            self.num_classes, dtype=torch.float32, device=device)
        static_weight = self.diw_static_class_weight if self.diw_static_class_weight is not None else torch.ones(
            self.num_classes, dtype=torch.float32, device=device)

        class_names = self._get_diw_class_names()
        alpha, beta, gamma = self._get_diw_exponents()
        weight_min, weight_max = self._get_diw_clip_range()
        norm_factor = 1.0
        if self.diw_static_norm_factor is not None:
            norm_factor = float(self.diw_static_norm_factor.detach().cpu())

        self._diw_log(
            '[DIW] version={} alpha_global={:.3f} beta_intra_domain={:.3f} gamma_image={:.3f} '
            'clip=[{:.3f}, {:.3f}] normalize={} static_norm_factor={:.4f}'.format(
                self.diw_version, alpha, beta, gamma, weight_min, weight_max,
                self.diw_normalize_weight, norm_factor)
        )
        self._diw_log('[DIW] domain_label mapping used by dataset: 0=external_RGB, 1=internal_B.')

        for cls_id in range(self.num_classes):
            domain_text = 'unknown'
            if self.diw_class_to_domain is not None:
                domain_id = int(self.diw_class_to_domain[cls_id].item())
                domain_text = DIW_DEFAULT_DOMAIN_NAMES.get(domain_id, f'domain_{domain_id}') if domain_id >= 0 else 'unknown'
            self._diw_log(
                '[DIW] class {:>2d} {:<20s} domain={:<12s} global={:.4f} intra={:.4f} combined_static={:.4f}'.format(
                    cls_id, class_names[cls_id], domain_text,
                    float(global_weight[cls_id].detach().cpu()),
                    float(intra_weight[cls_id].detach().cpu()),
                    float(static_weight[cls_id].detach().cpu()))
            )

        for group_name, class_ids in DIW_DEFAULT_GROUPS.items():
            valid_ids = [idx for idx in class_ids if idx < self.num_classes]
            if not valid_ids:
                continue
            group_mean = static_weight[valid_ids].mean()
            self._diw_log('[DIW] {} mean combined_static={:.4f}'.format(
                group_name, float(group_mean.detach().cpu())))

    def _configure_matcher_diw_cost(self):
        if not hasattr(self.matcher, 'set_diw_cost_weight'):
            if self.diw_apply_to_matcher_cost:
                _logger.warning(
                    "diw_apply_to_matcher_cost=True but matcher does not expose set_diw_cost_weight; matcher DIW is skipped.")
            return

        enabled = self.use_diw_loss and self.diw_apply_to_matcher_cost and self.diw_static_class_weight is not None
        self.matcher.set_diw_cost_weight(enabled=enabled, class_weight=self.diw_static_class_weight)

    @staticmethod
    def _is_classification_loss(loss):
        return loss in ('focal', 'vfl', 'mal')

    @staticmethod
    def _get_target_domain_label(target, device):
        domain_label = target.get('domain_label', 0)
        if torch.is_tensor(domain_label):
            if domain_label.numel() == 0:
                return torch.zeros((), dtype=torch.long, device=device)
            return domain_label.to(device=device, dtype=torch.long).reshape(-1)[0]
        return torch.tensor(int(domain_label), dtype=torch.long, device=device)

    def _get_local_diw_weights(self, targets, indices, device, dtype):
        local_weights = []
        for target, (_, target_idx) in zip(targets, indices):
            if target_idx.numel() == 0:
                continue

            target_idx = target_idx.to(device=device)
            labels = target['labels'].to(device=device, dtype=torch.long)
            matched_labels = labels[target_idx]
            if labels.numel() == 0:
                local_weights.append(torch.ones_like(matched_labels, dtype=dtype))
                continue

            unique_labels, label_counts = torch.unique(labels, return_counts=True)
            valid_labels = (unique_labels >= 0) & (unique_labels < self.num_classes)
            if not valid_labels.any():
                local_weights.append(torch.ones_like(matched_labels, dtype=dtype))
                continue

            count_map = torch.ones(self.num_classes, dtype=dtype, device=device)
            valid_unique = unique_labels[valid_labels]
            valid_counts = label_counts[valid_labels].to(device=device, dtype=dtype)
            count_map[valid_unique] = valid_counts
            mean_count = valid_counts.mean()

            image_weights = torch.ones_like(matched_labels, dtype=dtype)
            valid_matched = (matched_labels >= 0) & (matched_labels < self.num_classes)
            if valid_matched.any():
                image_weights[valid_matched] = mean_count / count_map[matched_labels[valid_matched]]
            local_weights.append(image_weights)

        if len(local_weights) == 0:
            return None
        return torch.cat(local_weights, dim=0)

    def _get_matched_domain_labels(self, targets, indices, device):
        labels = [
            self._get_target_domain_label(target, device).expand(target_idx.numel())
            for target, (_, target_idx) in zip(targets, indices)
            if target_idx.numel() > 0
        ]
        if len(labels) == 0:
            return None
        return torch.cat(labels, dim=0)

    def _get_intra_domain_weights(self, target_classes_o, domain_labels, device, dtype):
        if self.diw_domain_weight is None:
            return None

        domain_weight = self.diw_domain_weight.to(device=device, dtype=dtype)
        domain_weights = torch.ones_like(target_classes_o, dtype=dtype)
        base_valid = (
            (domain_labels >= 0) &
            (domain_labels < domain_weight.shape[0]) &
            (target_classes_o >= 0) &
            (target_classes_o < domain_weight.shape[1])
        )
        if not base_valid.any():
            return domain_weights

        use_target_domain = base_valid
        if self.diw_domain_count_mask is not None:
            domain_mask = self.diw_domain_count_mask.to(device=device)
            has_class_in_domain = torch.zeros_like(base_valid)
            has_class_in_domain[base_valid] = domain_mask[domain_labels[base_valid], target_classes_o[base_valid]]
            use_target_domain = base_valid & has_class_in_domain

        if use_target_domain.any():
            domain_weights[use_target_domain] = domain_weight[
                domain_labels[use_target_domain], target_classes_o[use_target_domain]]

        fallback = base_valid & ~use_target_domain
        if fallback.any() and self.diw_class_to_domain is not None:
            class_to_domain = self.diw_class_to_domain.to(device=device)
            fallback_domains = class_to_domain[target_classes_o[fallback]]
            fallback_valid = fallback_domains >= 0
            if fallback_valid.any():
                fallback_indices = torch.nonzero(fallback, as_tuple=False).flatten()[fallback_valid]
                domain_weights[fallback_indices] = domain_weight[
                    fallback_domains[fallback_valid], target_classes_o[fallback_indices]]

        return domain_weights

    def _log_diw_image_weight_range(self, image_weights, matched_weights):
        if (
            not self.diw_log_weight_stats or
            self.diw_log_first_n_batches <= 0 or
            self._diw_image_log_count >= self.diw_log_first_n_batches
        ):
            return
        if image_weights is None or image_weights.numel() == 0:
            return

        self._diw_image_log_count += 1
        image_min = float(image_weights.detach().min().cpu())
        image_max = float(image_weights.detach().max().cpu())
        final_min = float(matched_weights.detach().min().cpu())
        final_max = float(matched_weights.detach().max().cpu())
        self._diw_log(
            '[DIW] image-aware batch {:d}/{:d}: image_weight_range=[{:.4f}, {:.4f}] '
            'final_weight_range=[{:.4f}, {:.4f}]'.format(
                self._diw_image_log_count, self.diw_log_first_n_batches,
                image_min, image_max, final_min, final_max)
        )

    def _get_diw_weights(self, outputs, targets, indices, stage='main'):
        if not self.use_diw_loss or not self.diw_has_stats or 'pred_logits' not in outputs:
            return None

        src_logits = outputs['pred_logits']
        device, dtype = src_logits.device, src_logits.dtype
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return None
        batch_idx, src_idx = (idx[0].to(device=device), idx[1].to(device=device))

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(
            device=device, dtype=torch.long)
        if target_classes_o.numel() == 0:
            return None

        matched_weights = torch.ones_like(target_classes_o, dtype=dtype)
        alpha, beta, gamma = self._get_diw_exponents()

        if self.diw_global_weight is not None:
            global_weight = self.diw_global_weight.to(device=device, dtype=dtype)
            valid = (target_classes_o >= 0) & (target_classes_o < global_weight.numel())
            if valid.any():
                global_weights = torch.ones_like(matched_weights)
                global_weights[valid] = global_weight[target_classes_o[valid]]
                matched_weights = matched_weights * global_weights.pow(alpha)

        if self.diw_domain_weight is not None:
            domain_labels = self._get_matched_domain_labels(targets, indices, device)
            if domain_labels is not None:
                domain_weights = self._get_intra_domain_weights(target_classes_o, domain_labels, device, dtype)
                if domain_weights is not None:
                    matched_weights = matched_weights * domain_weights.pow(beta)

        image_weights = None
        if gamma != 0:
            image_weights = self._get_local_diw_weights(targets, indices, device, dtype)
            if image_weights is not None:
                matched_weights = matched_weights * image_weights.pow(gamma)

        weight_min, weight_max = self._get_diw_clip_range()
        matched_weights = matched_weights.clamp(min=weight_min, max=weight_max)
        if (
            self.diw_version == 'v2' and
            self.diw_normalize_weight and
            self.diw_static_norm_factor is not None
        ):
            norm_factor = self.diw_static_norm_factor.to(device=device, dtype=dtype).clamp_min(1e-12)
            matched_weights = (matched_weights / norm_factor).clamp(min=weight_min, max=weight_max)

        if self.diw_version == 'v2' and stage == 'main':
            self._log_diw_image_weight_range(image_weights, matched_weights)

        diw_weights = torch.ones_like(src_logits)
        valid_classes = (target_classes_o >= 0) & (target_classes_o < self.num_classes)
        if valid_classes.any():
            diw_weights[batch_idx[valid_classes], src_idx[valid_classes], target_classes_o[valid_classes]] = \
                matched_weights[valid_classes]
        return diw_weights

    def _add_diw_loss_meta(self, meta, loss, outputs, targets, indices, apply_diw=True, stage='main'):
        if not apply_diw:
            return meta
        if not self._is_classification_loss(loss):
            return meta

        diw_weights = self._get_diw_weights(outputs, targets, indices, stage=stage)
        if diw_weights is None:
            return meta

        meta = dict(meta)
        meta['diw_weights'] = diw_weights
        return meta

    def loss_distillation(self, outputs, targets, indices, num_boxes, **kwargs):
        student_feature_map = outputs.get('student_distill_output')
        teacher_feature_map = outputs.get('teacher_encoder_output')

        if student_feature_map is None or teacher_feature_map is None:
            return {'loss_distill': torch.tensor(0.0,
                                                 device=student_feature_map.device if student_feature_map is not None else torch.device(
                                                     'cuda'), requires_grad=True)}

        # _logger.info(f"[RTv4Criterion] Student feature map shape: {student_feature_map.shape}")
        # _logger.info(f"[RTv4Criterion] Teacher feature map shape: {teacher_feature_map.shape}")

        if student_feature_map.shape[1] != teacher_feature_map.shape[1]:
            _logger.error(
                f"[RTv4Criterion] Feature dimension mismatch! Student: {student_feature_map.shape[1]}, Teacher: {teacher_feature_map.shape[1]}")
            raise ValueError("Feature dimension mismatch between student and teacher for distillation loss.")

        H_s, W_s = student_feature_map.shape[2:]
        H_t, W_t = teacher_feature_map.shape[2:]

        target_h, target_w = H_s, W_s

        if (H_s, W_s) != (H_t, W_t):
            _logger.warning(
                f"[RTv4Criterion] Resizing teacher feature map from {H_t}x{W_t} to student's {H_s}x{W_s} for distillation.")
            teacher_feature_map = F.interpolate(teacher_feature_map,
                                                size=(target_h, target_w),
                                                mode='bilinear',
                                                align_corners=False)

        student_output_flat = student_feature_map.flatten(2).permute(0, 2, 1)
        teacher_output_flat = teacher_feature_map.flatten(2).permute(0, 2, 1)

        student_output_norm = F.normalize(student_output_flat, p=2, dim=-1)
        teacher_output_norm = F.normalize(teacher_output_flat, p=2, dim=-1)

        cos_sim = F.cosine_similarity(student_output_norm, teacher_output_norm, dim=-1)
        loss_distill = (1 - cos_sim).mean()

        return {'loss_distill': loss_distill}


    def _get_distillation_weight_for_epoch(self) -> float:
        fixed_weight = self.weight_dict.get('loss_distill', 0.0)
        return fixed_weight

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, diw_weights=None):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        if diw_weights is not None:
            loss = loss * diw_weights
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, diw_weights=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        if diw_weights is not None:
            loss = loss * diw_weights
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None, diw_weights=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        if diw_weights is not None:
            loss = loss * diw_weights
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou( \
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                    self.fgl_targets_dn = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                    self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                     self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou( \
                box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max + 1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max + 1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                                                                          (F.log_softmax(pred_corners / T, dim=1),
                                                                           F.softmax(target_corners.detach() / T,
                                                                                     dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, (
                                    (~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (
                                self.num_pos + self.num_neg)

        return losses

    def loss_domain(self, outputs, targets, indices, num_boxes, **kwargs):
        if "domain_logits" not in outputs:
            return {}

        # targets 是 list，需要整理成 batch tensor
        domain_labels = torch.tensor(
            [int(t.get("domain_label", 0)) for t in targets],
            device=outputs["domain_logits"].device,
            dtype=torch.long,
        )
        losses = {}
        loss = F.cross_entropy(outputs["domain_logits"], domain_labels)
        losses["loss_domain"] = loss

        return losses

    def loss_confusion(self, outputs, targets, indices, num_boxes, **kwargs):
        proto_logits = outputs.get('confusion_proto_logits')
        if proto_logits is None or indices is None:
            return {}

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_confusion': proto_logits.sum() * 0.0}

        device = proto_logits.device
        batch_idx, src_idx = idx[0].to(device=device), idx[1].to(device=device)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(
            device=device, dtype=torch.long)
        if target_classes_o.numel() == 0:
            return {'loss_confusion': proto_logits.sum() * 0.0}

        valid = (target_classes_o >= 0) & (target_classes_o < self.num_classes)
        if not valid.any():
            return {'loss_confusion': proto_logits.sum() * 0.0}

        matched_logits = proto_logits[batch_idx[valid], src_idx[valid]]
        target_classes_o = target_classes_o[valid]
        relation = self.confusion_relation.to(device=device)
        hard_mask = relation[target_classes_o]
        has_hard = hard_mask.any(dim=-1)
        if not has_hard.any():
            return {'loss_confusion': proto_logits.sum() * 0.0}

        matched_logits = matched_logits[has_hard]
        target_classes_o = target_classes_o[has_hard]
        hard_mask = hard_mask[has_hard]

        pos_logits = matched_logits.gather(1, target_classes_o.unsqueeze(1))
        hard_logits = matched_logits.masked_fill(~hard_mask, -torch.inf)
        topk = min(self.confusion_topk, hard_logits.shape[-1])
        hard_values, hard_indices = torch.topk(hard_logits, k=topk, dim=-1)
        finite = torch.isfinite(hard_values)
        if not finite.any():
            return {'loss_confusion': proto_logits.sum() * 0.0}

        pair_margins = torch.full_like(hard_values, self.confusion_margin)
        if self.confusion_prior_mode == 'margin':
            prior_rows = self.confusion_prior.to(device=device, dtype=matched_logits.dtype)[target_classes_o]
            selected_prior = prior_rows.gather(1, hard_indices)
            pair_margins = pair_margins * (1.0 + self.confusion_prior_strength * selected_prior)

        margins = F.relu(pair_margins + hard_values - pos_logits)
        per_query = (margins * finite.to(margins.dtype)).sum(dim=-1) / finite.sum(dim=-1).clamp_min(1)
        loss = per_query[finite.any(dim=-1)].sum() / num_boxes
        return {'loss_confusion': loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                       for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
            'distill': self.loss_distillation,  # NEW: Add distillation loss
            'domain': self.loss_domain,
            'confusion': self.loss_confusion,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss_name in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            if loss_name == 'distill':
                l_dict = self.get_loss(loss_name, outputs, targets, None, None, **kwargs)
                if 'loss_distill' in l_dict and l_dict['loss_distill'] != 0:
                    dynamic_weight = self._get_distillation_weight_for_epoch()
                    l_dict['loss_distill'] = l_dict['loss_distill'] * dynamic_weight
                losses.update(l_dict)
            else:
                use_uni_set = self.use_uni_set and (loss_name in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else indices
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss_name, outputs, targets, indices_in)
                meta = self._add_diw_loss_meta(
                    meta, loss_name, outputs, targets, indices_in,
                    apply_diw=self.diw_apply_to_main_cls, stage='main')
                l_dict = self.get_loss(loss_name, outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:  # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    meta = self._add_diw_loss_meta(
                        meta, loss, aux_outputs, targets, indices_in,
                        apply_diw=self.diw_apply_to_aux_cls, stage='aux')
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                meta = self._add_diw_loss_meta(
                    meta, loss, aux_outputs, targets, indices_in,
                    apply_diw=self.diw_apply_to_aux_cls, stage='aux')
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    meta = self._add_diw_loss_meta(
                        meta, loss, aux_outputs, enc_targets, indices_in,
                        apply_diw=self.diw_apply_to_encoder_loss, stage='encoder')
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:  # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou( \
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes',):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                                         torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum',
                                         avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
               + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
