"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from ..core import register

import logging
_logger = logging.getLogger(__name__)

__all__ = ['HybridEncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# TODO, add activation for cv1 following YOLOv10
# self.cv1 = Conv(c1, c2, 1, 1)
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MultiScaleGate(nn.Module):
    def __init__(self, hidden_dim, gate_dim=None):
        super().__init__()
        gate_dim = hidden_dim if gate_dim is None else gate_dim
        self.gate = nn.Sequential(
            nn.Conv2d(gate_dim, hidden_dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, feat, gate_feat):
        return feat * (1.0 + self.gate(gate_feat))


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', 'num_classes']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 num_classes=80,
                 version='dfine',
                 distill_teacher_dim=0,
                 use_ms_gated_aifi=False,
                 pmgi_hidden_dim=None,
                 use_dcp_aifi=False,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.num_classes = num_classes
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.distill_teacher_dim = distill_teacher_dim
        self.use_ms_gated_aifi = use_ms_gated_aifi and len(in_channels) >= 3 and 2 in use_encoder_idx
        self.pmgi_hidden_dim = hidden_dim if pmgi_hidden_dim is None else pmgi_hidden_dim
        self.use_dcp_aifi = use_dcp_aifi

        assert len(use_encoder_idx) > 0, "use_encoder_idx must specify at least one encoder output"
        if self.use_dcp_aifi:
            assert len(in_channels) >= 3 and 2 in use_encoder_idx, \
                "DCP-AIFI requires S5 (encoder index 2) in use_encoder_idx."
            assert not self.use_ms_gated_aifi, \
                "DCP-AIFI only modifies the S5 AIFI path. Disable use_ms_gated_aifi when use_dcp_aifi=True."
        # target AIFI output F5 for distillation
        self.encoder_idx_for_distillation = use_encoder_idx[-1]

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))

            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        if self.use_ms_gated_aifi:
            pmgi_encoder_layer = TransformerEncoderLayer(
                self.pmgi_hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=enc_act
            )
            self.ms_pool_proj = nn.ModuleList([
                ConvNormLayer_fuse(hidden_dim, self.pmgi_hidden_dim, 1, 1, act=act),
                ConvNormLayer_fuse(hidden_dim, self.pmgi_hidden_dim, 1, 1, act=act),
                ConvNormLayer_fuse(hidden_dim, self.pmgi_hidden_dim, 1, 1, act=act),
            ])
            self.ms_encoder = TransformerEncoder(copy.deepcopy(pmgi_encoder_layer), num_encoder_layers)
            self.ms_level_embed = nn.Parameter(torch.zeros(3, 1, 1, self.pmgi_hidden_dim))
            self.ms_gate = nn.ModuleList([MultiScaleGate(hidden_dim, self.pmgi_hidden_dim) for _ in range(3)])
            self.ms_smooth_p4 = ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 1, act=act)
            self.ms_smooth_p3_stage1 = ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 1, act=act)
            self.ms_smooth_p3_stage2 = ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 1, act=act)

        if self.use_dcp_aifi:
            self.class_prototypes = nn.Parameter(torch.empty(num_classes, hidden_dim))
            self.dcp_domain_conditioner = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                get_activation(act),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
            self.dcp_proto_norm = nn.LayerNorm(hidden_dim)
            self.dcp_proto_residual = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                get_activation(act),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # feature_projector
        self.feature_projector = None
        if self.distill_teacher_dim > 0:
            self.feature_projector = nn.Sequential(
                    nn.Linear(hidden_dim, self.distill_teacher_dim),
                    # nn.GELU(),
                )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # TODO, add activation for those lateral convs
            if version == 'dfine':
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
                if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if hasattr(self, 'class_prototypes'):
            nn.init.normal_(self.class_prototypes, std=0.02)
            nn.init.zeros_(self.dcp_domain_conditioner[-1].weight)
            nn.init.zeros_(self.dcp_domain_conditioner[-1].bias)
            nn.init.zeros_(self.dcp_proto_residual[-1].weight)
            nn.init.zeros_(self.dcp_proto_residual[-1].bias)

        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def _get_dcp_class_tokens(self, domain_emb, fallback_feat):
        bs, c = fallback_feat.shape[:2]
        proto = self.class_prototypes.unsqueeze(0).expand(bs, -1, -1)
        proto = proto.to(device=fallback_feat.device, dtype=fallback_feat.dtype)

        if domain_emb is None:
            domain_emb = F.adaptive_avg_pool2d(fallback_feat, 1).flatten(1)
        domain_emb = domain_emb.to(device=fallback_feat.device, dtype=fallback_feat.dtype)
        if domain_emb.shape[-1] != c:
            raise ValueError(
                f"DCP-AIFI domain embedding dim must be {c}, got {domain_emb.shape[-1]}."
            )

        gamma, beta = self.dcp_domain_conditioner(domain_emb).unsqueeze(1).chunk(2, dim=-1)
        conditioned_proto = self.dcp_proto_norm(proto) * (1.0 + torch.tanh(gamma)) + beta
        return conditioned_proto + self.dcp_proto_residual(conditioned_proto)

    def forward(self, feats, domain_emb=None):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        distill_student_output = None

        # encoder
        if self.num_encoder_layers > 0:
            if self.use_ms_gated_aifi:
                s3, s4, s5 = proj_feats[0], proj_feats[1], proj_feats[2]
                s3_pool2 = F.avg_pool2d(F.avg_pool2d(s3, kernel_size=2, stride=2), kernel_size=2, stride=2)
                s4_pool = F.avg_pool2d(s4, kernel_size=2, stride=2)
                pooled_base_feats = [s3_pool2, s4_pool, s5]

                pooled_feats = [
                    self.ms_pool_proj[0](pooled_base_feats[0]),
                    self.ms_pool_proj[1](pooled_base_feats[1]),
                    self.ms_pool_proj[2](pooled_base_feats[2]),
                ]
                h5, w5 = pooled_feats[-1].shape[2:]
                token_groups = [feat.flatten(2).permute(0, 2, 1) for feat in pooled_feats]

                base_pos_embed = self.build_2d_sincos_position_embedding(
                    w5, h5, self.pmgi_hidden_dim, self.pe_temperature).to(s5.device)

                level_tokens = []
                pos_tokens = []
                for level_idx, feat_tokens in enumerate(token_groups):
                    level_tokens.append(feat_tokens + self.ms_level_embed[level_idx])
                    pos_tokens.append(base_pos_embed)

                src_flatten = torch.cat(level_tokens, dim=1)
                pos_embed = torch.cat(pos_tokens, dim=1)
                memory: torch.Tensor = self.ms_encoder(src_flatten, src_mask=None, pos_embed=pos_embed)

                split_sizes = [tokens.shape[1] for tokens in token_groups]
                memory_groups = torch.split(memory, split_sizes, dim=1)
                gated_feats = []
                for feat, memory_tokens, gate in zip(pooled_base_feats, memory_groups, self.ms_gate):
                    memory_feat = memory_tokens.permute(0, 2, 1).reshape(-1, self.pmgi_hidden_dim, h5, w5).contiguous()
                    gated_feats.append(gate(feat, memory_feat))

                f3_pool2, f4_pool, f5 = gated_feats

                f4 = F.interpolate(f4_pool, size=proj_feats[1].shape[2:], mode='nearest')
                f4 = self.ms_smooth_p4(f4)

                f3 = F.interpolate(f3_pool2, size=proj_feats[1].shape[2:], mode='nearest')
                f3 = self.ms_smooth_p3_stage1(f3)
                f3 = F.interpolate(f3, size=proj_feats[0].shape[2:], mode='nearest')
                f3 = self.ms_smooth_p3_stage2(f3)

                proj_feats[0], proj_feats[1], proj_feats[2] = f3, f4, f5

                if self.training and self.feature_projector is not None:
                    distill_student_output = self.feature_projector(
                        f5.permute(0, 2, 3, 1)
                    ).permute(0, 3, 1, 2)
            else:
                for i, enc_ind in enumerate(self.use_encoder_idx):
                    h, w = proj_feats[enc_ind].shape[2:]
                    # flatten [B, C, H, W] to [B, HxW, C]
                    src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                    if self.training or self.eval_spatial_size is None:
                        pos_embed = self.build_2d_sincos_position_embedding(
                            w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                    else:
                        pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                    if self.use_dcp_aifi and enc_ind == 2:
                        class_tokens = self._get_dcp_class_tokens(domain_emb, proj_feats[enc_ind])
                        class_pos_embed = torch.zeros(
                            1, self.num_classes, self.hidden_dim,
                            device=src_flatten.device,
                            dtype=src_flatten.dtype
                        )
                        memory_input = torch.cat([src_flatten, class_tokens], dim=1)
                        pos_embed_input = torch.cat([pos_embed, class_pos_embed], dim=1)
                        memory :torch.Tensor = self.encoder[i](
                            memory_input, src_mask=None, pos_embed=pos_embed_input
                        )
                        memory = memory[:, :h * w]
                    else:
                        memory :torch.Tensor = self.encoder[i](src_flatten, src_mask=None, pos_embed=pos_embed)

                    # Reshape back to [B, C, H, W] for subsequent FPN/PAN layers
                    proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

                    # Apply feature projector to F5
                    if self.training and self.feature_projector is not None and enc_ind == self.encoder_idx_for_distillation:
                        # _logger.info(f"[HybridEncoder] Feature size: {h}x{w}")
                        distill_student_output = self.feature_projector(proj_feats[enc_ind].permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # [B, distill_teacher_dim, H, W]


        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        if self.training and distill_student_output is not None:
            return outs, distill_student_output
        return outs
