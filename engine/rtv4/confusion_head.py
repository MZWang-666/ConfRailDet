import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_RAIL_DOMAIN_CLASS_IDS = (
    (0, 1, 2, 3, 4),  # internal_B
    (5, 6, 7, 8),     # external_RGB
)


def normalize_domain_class_ids(domain_class_ids, num_classes):
    if domain_class_ids is None:
        if num_classes == 9:
            domain_class_ids = DEFAULT_RAIL_DOMAIN_CLASS_IDS
        else:
            domain_class_ids = (tuple(range(num_classes)),)

    if isinstance(domain_class_ids, dict):
        groups = domain_class_ids.values()
    else:
        groups = domain_class_ids

    normalized = []
    for group in groups:
        valid = sorted({int(cls_id) for cls_id in group if 0 <= int(cls_id) < num_classes})
        if len(valid) > 0:
            normalized.append(tuple(valid))
    return tuple(normalized)


def build_confusion_relation(num_classes, domain_class_ids=None):
    relation = torch.zeros(num_classes, num_classes, dtype=torch.bool)
    for class_ids in normalize_domain_class_ids(domain_class_ids, num_classes):
        idx = torch.as_tensor(class_ids, dtype=torch.long)
        relation[idx[:, None], idx[None, :]] = True
    relation.fill_diagonal_(False)
    return relation


class DomainGuidedConfusionHead(nn.Module):
    def __init__(
        self,
        num_classes=9,
        hidden_dim=256,
        domain_class_ids=None,
        temperature=0.2,
        logit_alpha=0.25,
        fusion_mode='add',
        use_domain_context=False,
        context_hidden_dim=None,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if fusion_mode not in ('add', 'weighted_sum'):
            raise ValueError("fusion_mode must be 'add' or 'weighted_sum'.")

        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.logit_alpha = float(logit_alpha)
        self.fusion_mode = fusion_mode
        self.use_domain_context = bool(use_domain_context)

        self.class_prototypes = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        if self.use_domain_context:
            context_hidden_dim = int(context_hidden_dim or self.hidden_dim)
            self.context_encoder = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, context_hidden_dim),
                nn.SiLU(),
                nn.Linear(context_hidden_dim, self.hidden_dim * 2),
            )
        else:
            self.context_encoder = None

        self.register_buffer(
            'confusion_relation',
            build_confusion_relation(self.num_classes, domain_class_ids),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.class_prototypes, std=0.02)
        nn.init.xavier_uniform_(self.query_proj[1].weight)
        nn.init.zeros_(self.query_proj[1].bias)
        if self.context_encoder is not None:
            nn.init.zeros_(self.context_encoder[-1].weight)
            nn.init.zeros_(self.context_encoder[-1].bias)

    def _get_contextual_prototypes(self, domain_context):
        prototypes = self.class_prototypes
        if self.context_encoder is None or domain_context is None:
            return prototypes.unsqueeze(0)

        gamma, beta = self.context_encoder(domain_context).chunk(2, dim=-1)
        prototypes = prototypes.unsqueeze(0)
        return prototypes * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, query_features, class_logits, domain_context=None):
        query_embed = F.normalize(self.query_proj(query_features), dim=-1)
        prototypes = F.normalize(self._get_contextual_prototypes(domain_context), dim=-1)

        if prototypes.shape[0] == 1:
            proto_logits = torch.matmul(query_embed, prototypes.squeeze(0).transpose(0, 1))
        else:
            proto_logits = torch.einsum('bqd,bcd->bqc', query_embed, prototypes)
        proto_logits = proto_logits / self.temperature

        if self.fusion_mode == 'weighted_sum':
            fused_logits = (1.0 - self.logit_alpha) * class_logits + self.logit_alpha * proto_logits
        else:
            fused_logits = class_logits + self.logit_alpha * proto_logits

        return fused_logits, proto_logits
