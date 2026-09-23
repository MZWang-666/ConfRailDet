import csv
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig
from engine.rtv4.confusion_head import DomainGuidedConfusionHead, build_confusion_relation
from engine.rtv4.rtv4_criterion import RTv4Criterion


def main():
    config_path = ROOT / "configs" / "confraildet" / "confraildet_rtv4_hgnetv2_s.yml"
    cfg = YAMLConfig(
        str(config_path),
        HGNetv2={"pretrained": False},
        RTv4Criterion={"confusion_prior_mode": "none"},
    )
    assert cfg.yaml_cfg["num_classes"] == 9

    groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
    relation = build_confusion_relation(9, groups)
    assert relation.shape == (9, 9)
    assert relation[2, 3] and relation[5, 8]
    assert not relation[0, 5] and not relation.diagonal().any()

    head = DomainGuidedConfusionHead(
        num_classes=9,
        hidden_dim=32,
        domain_class_ids=groups,
        temperature=0.2,
        logit_alpha=0.25,
    )
    query = torch.randn(2, 12, 32)
    native_logits = torch.randn(2, 12, 9)
    fused_logits, prototype_logits = head(query, native_logits)
    assert fused_logits.shape == native_logits.shape
    assert prototype_logits.shape == native_logits.shape
    assert torch.isfinite(fused_logits).all()

    criterion = RTv4Criterion(
        matcher=None,
        weight_dict={"loss_confusion": 0.2},
        losses=["confusion"],
        num_classes=9,
        confusion_domain_class_ids=groups,
        confusion_margin=0.2,
        confusion_topk=2,
        confusion_prior_mode="none",
    )
    outputs = {"confusion_proto_logits": prototype_logits}
    targets = [
        {"labels": torch.tensor([3], dtype=torch.long)},
        {"labels": torch.tensor([7], dtype=torch.long)},
    ]
    indices = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1]), torch.tensor([0])),
    ]
    loss = criterion.loss_confusion(outputs, targets, indices, num_boxes=2.0)["loss_confusion"]
    assert loss.ndim == 0 and torch.isfinite(loss)

    with tempfile.TemporaryDirectory() as temp_dir:
        prior_path = Path(temp_dir) / "confusion_prior_matrix.csv"
        with prior_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class", *range(9)])
            for row in range(9):
                writer.writerow([row, *[0.5 if relation[row, col] else 0.0 for col in range(9)]])
        cpm_criterion = RTv4Criterion(
            matcher=None,
            weight_dict={"loss_confusion": 0.2},
            losses=["confusion"],
            num_classes=9,
            confusion_domain_class_ids=groups,
            confusion_prior_mode="margin",
            confusion_prior_path=str(prior_path),
            confusion_prior_strength=1.0,
        )
        assert cpm_criterion.confusion_prior.shape == (9, 9)
        assert cpm_criterion.confusion_prior[2, 3] == 0.5
        assert cpm_criterion.confusion_prior[0, 5] == 0.0

    model = cfg.model.eval()
    with torch.inference_mode():
        model_outputs = model(torch.randn(1, 3, 640, 640))
    assert model_outputs["pred_logits"].shape == (1, 300, 9)
    assert model_outputs["pred_boxes"].shape == (1, 300, 4)
    assert model_outputs["confusion_proto_logits"].shape == (1, 300, 9)
    assert torch.isfinite(model_outputs["pred_logits"]).all()

    print(
        "ConfRailDet smoke test passed: config, PSF, relation mask, DHCM loss, "
        "CPM loading, model construction, and data-free forward are valid."
    )


if __name__ == "__main__":
    main()
