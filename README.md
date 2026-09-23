# ConfRailDet

Official code preparation for **ConfRailDet: Domain-Guided Fine-Grained Confusion Modeling for Unified Rail Defect Detection**.

ConfRailDet addresses fine-grained inter-class confusion in a unified detector for heterogeneous railway inspection images. Internal ultrasonic B-scan categories and external RGB categories are domain-exclusive. Domain information is therefore used to restrict valid confusion relations within each acquisition domain, rather than to align the same category across domains.

## Method Overview

ConfRailDet extends an RT-DETRv4-S detector with three components:

- **Prototype Similarity Fusion (PSF):** learns class prototypes from decoder query features and fuses prototype-similarity logits with the native classification logits during training and inference.
- **Domain-Restricted Hard-Confusion Margin (DHCM):** selects hard negative classes only from the same acquisition-domain category set and imposes a query-level margin objective.
- **Confusion-Prior Margin Modulation (CPM):** constructs a fixed class-pair prior from training-set prototype similarity and baseline training confusion, then uses it to modulate pair-specific DHCM margins.

The default category groups are `[0, 1, 2, 3, 4]` for internal ultrasonic B-scan inspection and `[5, 6, 7, 8]` for external RGB inspection.

## Installation

Python 3.10 or newer is recommended. Install a PyTorch build compatible with your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

No dataset, annotation file, pretrained weight, training output, or TensorRT artifact is included in this repository.

## Data Preparation

The example configuration expects COCO-format annotations and the following local structure:

```text
data/
└── UHRD-9/
    ├── annotations/
    │   ├── train.json
    │   └── val.json
    ├── train/
    │   └── images/
    └── val/
        └── images/
```

If a different layout is used, edit `configs/dataset/uhrd9_detection.example.yml`. Do not commit private images or annotations.

**Data availability.** The UHRD-9 dataset is not publicly available due to confidentiality and proprietary restrictions associated with the railway inspection systems. Data may be available from the corresponding author upon reasonable request and subject to approval from the data provider.

## Confusion-Prior Construction

CPM uses only the training split. First train an RT-DETRv4-S baseline or provide a compatible baseline checkpoint. Then compute training-set confusion statistics:

```bash
python tools/analysis/confusion_matrix_eval.py \
  --config configs/confraildet/rtdetrv4_s_baseline.yml \
  --checkpoint checkpoints/rtdetrv4_s_baseline.pth \
  --split train
```

Extract training-set visual prototypes and combine their similarity with the baseline training confusion:

```bash
python tools/analysis/prototype_similarity.py \
  --config configs/confraildet/rtdetrv4_s_baseline.yml \
  --checkpoint checkpoints/rtdetrv4_s_baseline.pth
```

The resulting prior is written to `outputs/analysis/train_prototype_similarity/confusion_prior_matrix.csv`. Validation annotations are not used to construct this prior.

## Training

Train the full ConfRailDet model with PSF, DHCM, and CPM:

```bash
python train_confraildet.py --ablation full --relation domain_all --use-amp
```

Available ablations can be listed with:

```bash
python train_confraildet.py --list-ablations
```

The baseline can be trained without a confusion prior:

```bash
python train_confraildet.py --ablation baseline --use-amp
```

## Evaluation and Inference

Evaluate a trained checkpoint on the configured validation split:

```bash
python train.py \
  -c configs/confraildet/confraildet_rtv4_hgnetv2_s.yml \
  -r checkpoints/confraildet.pth \
  --test-only
```

Run inference on an image or video:

```bash
python tools/inference/infer.py \
  -c configs/confraildet/confraildet_rtv4_hgnetv2_s.yml \
  -r checkpoints/confraildet.pth \
  -i path/to/input.jpg \
  -d cuda
```

Video inference additionally requires `opencv-python`.

## Repository Layout

```text
configs/                 Model, runtime, and example dataset configurations
engine/                  Detector, PSF head, DHCM/CPM criterion, and runtime code
scripts/smoke_test.py    Data-free component smoke test
tools/analysis/          Baseline confusion and prototype-prior construction
tools/inference/         Image and video inference
train.py                 Generic training and evaluation entry point
train_confraildet.py     ConfRailDet experiment and ablation entry point
```

## Citation

Before the associated paper is published, please cite this software repository:

```bibtex
@software{wang2026confraildet,
  author = {Mingzhong Wang and Zijun Wu and Jiancheng Long},
  title = {ConfRailDet: Domain-Guided Fine-Grained Confusion Modeling for Unified Rail Defect Detection},
  year = {2026},
  url = {https://github.com/MZWang-666/ConfRailDet},
  note = {GitHub repository}
}
```

The bibliographic record will be updated after publication. No article DOI has been assigned here.

## Acknowledgments

This implementation builds on [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) and [D-FINE](https://github.com/Peterande/D-FINE). Upstream copyright and license notices are retained in the corresponding source files.

## License

This repository is released under the Apache License 2.0. See `LICENSE` for details.
