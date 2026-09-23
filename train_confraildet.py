import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "configs/confraildet/confraildet_rtv4_hgnetv2_s.yml"
DEFAULT_PRIOR_PATH = "outputs/analysis/train_prototype_similarity/confusion_prior_matrix.csv"


ABLATION_PRESETS = {
    "full": {
        "description": "PSF + DHCM + CPM",
        "prior_mode": "margin",
        "output_dir": "./outputs/confraildet_full",
        "head": True,
        "alpha": 0.25,
        "confusion_loss": True,
    },
    "psf_dhcm": {
        "description": "PSF + DHCM without CPM",
        "prior_mode": "none",
        "output_dir": "./outputs/confraildet_psf_dhcm",
        "head": True,
        "alpha": 0.25,
        "confusion_loss": True,
    },
    "fusion_only": {
        "description": "PSF only",
        "prior_mode": "none",
        "output_dir": "./outputs/confraildet_psf_only",
        "head": True,
        "alpha": 0.25,
        "confusion_loss": False,
    },
    "dhcm_only": {
        "description": "DHCM only",
        "prior_mode": "none",
        "output_dir": "./outputs/confraildet_dhcm_only",
        "head": True,
        "alpha": 0.0,
        "confusion_loss": True,
    },
    "baseline": {
        "description": "RT-DETRv4-S baseline",
        "prior_mode": "none",
        "output_dir": "./outputs/rtdetrv4_s_baseline",
        "head": False,
        "alpha": 0.0,
        "confusion_loss": False,
    },
}


RELATION_PRESETS = {
    "domain_all": [[0, 1, 2, 3, 4], [5, 6, 7, 8]],
    "global_all": [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
    "external_only": [[5, 6, 7, 8]],
    "bolt_only": [[2, 3]],
}


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_command(args):
    preset = ABLATION_PRESETS[args.ablation]
    groups = RELATION_PRESETS[args.relation]
    prior_mode = args.prior_mode or preset["prior_mode"]
    losses = "[mal,boxes,local,confusion]" if preset["confusion_loss"] else "[mal,boxes,local]"
    loss_weight = 0.2 if preset["confusion_loss"] else 0.0
    output_dir = args.output_dir or preset["output_dir"]

    command = [
        sys.executable,
        "train.py",
        "-c",
        str(resolve_path(args.config)),
        f"--seed={args.seed}",
    ]
    if args.resume:
        command.extend(["--resume", args.resume])
    if args.tuning:
        command.extend(["--tuning", args.tuning])
    if args.test_only:
        command.append("--test-only")
    if args.use_amp:
        command.append("--use-amp")

    updates = [
        f"epoches={args.epochs}",
        f"train_dataloader.total_batch_size={args.batch_size}",
        f"val_dataloader.total_batch_size={args.batch_size}",
        f"output_dir={output_dir}",
        f"DFINETransformer.use_confusion_head={preset['head']}",
        f"DFINETransformer.confusion_logit_alpha={preset['alpha']}",
        "DFINETransformer.confusion_use_domain_context=False",
        f"DFINETransformer.confusion_domain_class_ids={groups}",
        f"RTv4Criterion.losses={losses}",
        f"RTv4Criterion.weight_dict.loss_confusion={loss_weight}",
        f"RTv4Criterion.confusion_domain_class_ids={groups}",
        f"RTv4Criterion.confusion_prior_mode={prior_mode}",
        f"RTv4Criterion.confusion_prior_path={resolve_path(args.prior_path).as_posix()}",
        f"RTv4Criterion.confusion_prior_strength={args.prior_strength}",
    ]
    if args.num_workers is not None:
        updates.extend([
            f"train_dataloader.num_workers={args.num_workers}",
            f"val_dataloader.num_workers={args.num_workers}",
        ])
    if args.extra_update:
        updates.extend(args.extra_update)
    command.extend(["-u", *updates])
    return command, prior_mode


def main(args):
    if args.list_ablations:
        for name, preset in ABLATION_PRESETS.items():
            print(f"{name}: {preset['description']}")
        return 0

    config = resolve_path(args.config)
    if not config.exists():
        raise FileNotFoundError(config)

    command, prior_mode = build_command(args)
    prior_path = resolve_path(args.prior_path)
    if prior_mode == "margin" and not prior_path.exists():
        raise FileNotFoundError(
            f"CPM prior not found: {prior_path}. Build it with "
            "tools/analysis/prototype_similarity.py before full training."
        )
    if args.prior_strength < 0:
        raise ValueError("--prior-strength must be non-negative")

    print(f"Experiment: {args.ablation} ({ABLATION_PRESETS[args.ablation]['description']})")
    print(f"Relation: {args.relation} -> {RELATION_PRESETS[args.relation]}")
    print(f"Prior mode: {prior_mode}")
    print("Command:")
    print(" ".join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return subprocess.call(command, cwd=ROOT, env=env)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConfRailDet and its ablations.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ablation", default="full", choices=sorted(ABLATION_PRESETS))
    parser.add_argument("--relation", default="domain_all", choices=sorted(RELATION_PRESETS))
    parser.add_argument("--prior-mode", choices=("none", "margin"), default=None)
    parser.add_argument("--prior-path", default=DEFAULT_PRIOR_PATH)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--tuning", default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--list-ablations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-u", "--extra-update", nargs="*", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
