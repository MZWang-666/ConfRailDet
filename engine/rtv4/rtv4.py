"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register
from .domain_head import DomainHead


__all__ = ['RTv4', ]


@register()
class RTv4(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        use_domain_branch=False,
        use_domain_film=True,
        domain_in_channels=None,
        domain_embed_dim=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.use_domain_branch = use_domain_branch
        self.use_domain_film = use_domain_branch and use_domain_film

        if self.use_domain_branch:
            encoder_in_channels = getattr(encoder, 'in_channels', None)
            encoder_hidden_dim = getattr(encoder, 'hidden_dim', None)
            if domain_in_channels is None:
                if encoder_in_channels is None:
                    raise ValueError("domain_in_channels must be set when encoder.in_channels is unavailable.")
                domain_in_channels = encoder_in_channels[-1]
            if domain_embed_dim is None:
                domain_embed_dim = encoder_hidden_dim if encoder_hidden_dim is not None else 256

            self.domain_head = DomainHead(in_channels=domain_in_channels, embed_dim=domain_embed_dim)
            if self.use_domain_film:
                film_dim = encoder_hidden_dim if encoder_hidden_dim is not None else domain_embed_dim
                self.film_gamma = nn.Linear(domain_embed_dim, film_dim)
                self.film_beta = nn.Linear(domain_embed_dim, film_dim)

    def forward(self, x, targets=None, teacher_encoder_output=None):
        x_backbone = self.backbone(x)  # [S3, S4, S5] features from backbone

        domain_emb, domain_logits = None, None
        if self.use_domain_branch:
            _, _, s5 = x_backbone
            domain_emb, domain_logits = self.domain_head(s5)

        if getattr(self.encoder, 'use_dcp_aifi', False):
            encoder_output = self.encoder(x_backbone, domain_emb=domain_emb)
        else:
            encoder_output = self.encoder(x_backbone)

        student_distill_output = None
        if self.training and isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            x_fpn_features, student_distill_output = encoder_output
        else:
            x_fpn_features = encoder_output

        if self.use_domain_film and domain_emb is not None:
            gamma = self.film_gamma(domain_emb).unsqueeze(-1).unsqueeze(-1)
            beta = self.film_beta(domain_emb).unsqueeze(-1).unsqueeze(-1)
            x_fpn_features = [feat * (1 + gamma) + beta for feat in x_fpn_features]

        x_decoder_out = self.decoder(x_fpn_features, targets)
        if self.training and domain_logits is not None:
            x_decoder_out['domain_logits'] = domain_logits

        if self.training and student_distill_output is not None and teacher_encoder_output is not None:
            x_decoder_out['student_distill_output'] = student_distill_output
            x_decoder_out['teacher_encoder_output'] = teacher_encoder_output

        return x_decoder_out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
