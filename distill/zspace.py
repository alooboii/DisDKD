import math
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from distill.rkd_losses import rkd_angle_loss, rkd_distance_loss


class LayerConditionedZSpace(nn.Module):
    """
    Layer-conditioned common Z-space with lightweight adapters.

    Supports hidden states from CNN [B, C, H, W] and transformer [B, N, C] tensors.
    """

    def __init__(
        self,
        layer_names: Iterable[str],
        z_dim: int = 128,
        z_tokens: int = 4,
        z_tokens_mode: str = "spatial",
        use_token_norm: bool = True,
        mlp_expansion: float = 1.0,
    ):
        super().__init__()
        self.layer_names = list(layer_names)
        self.layer_to_idx = {name: i for i, name in enumerate(self.layer_names)}
        self.z_dim = z_dim
        self.z_tokens = z_tokens
        self.z_tokens_mode = z_tokens_mode
        self.use_token_norm = use_token_norm
        self.mlp_expansion = mlp_expansion

        self.layer_embed = nn.Embedding(max(1, len(self.layer_names)), z_dim)

        self.encoder_mlps = nn.ModuleDict()
        self.decoder_mlps = nn.ModuleDict()

        # Transition alignment modules (lightweight, explicit).
        self.align_conv = nn.ModuleDict()
        self.align_linear = nn.ModuleDict()

        # Runtime metadata initialized from observed features.
        self.layer_input_channels: Dict[str, int] = {}
        self.layer_shape_templates: Dict[str, Tuple[int, ...]] = {}

    def _layer_key(self, layer_name: str) -> str:
        if layer_name not in self.layer_to_idx:
            raise ValueError(f"Unknown layer '{layer_name}'. Known: {self.layer_names}")
        return layer_name

    @staticmethod
    def _flatten_cnn_tokens(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert hidden state to token tensor [B, T, C]."""
        if x.dim() == 4:
            if self.z_tokens_mode == "global":
                pooled = F.adaptive_avg_pool2d(x, output_size=1)
                return self._flatten_cnn_tokens(pooled)

            if self.z_tokens_mode == "spatial":
                side = int(round(math.sqrt(self.z_tokens)))
                if side * side != self.z_tokens:
                    raise ValueError(
                        f"z_tokens={self.z_tokens} is not a perfect square for spatial mode"
                    )
                pooled = F.adaptive_avg_pool2d(x, output_size=(side, side))
                return self._flatten_cnn_tokens(pooled)

            raise ValueError(
                f"Unknown z_tokens_mode '{self.z_tokens_mode}'. Use global|spatial."
            )

        if x.dim() == 3:
            tokens = x
            if self.z_tokens > 0 and tokens.size(1) != self.z_tokens:
                tokens = tokens.transpose(1, 2)
                tokens = F.interpolate(
                    tokens, size=self.z_tokens, mode="linear", align_corners=False
                )
                tokens = tokens.transpose(1, 2)
            return tokens

        raise ValueError(f"Unsupported hidden state shape for tokenization: {tuple(x.shape)}")

    @staticmethod
    def _tokens_to_shape(tokens: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Project token tensor [B, T, C] back to target hidden shape."""
        bsz = tokens.size(0)

        if len(target_shape) == 3:
            c, h, w = target_shape
            t = tokens.size(1)
            c_tok = tokens.size(2)
            if c_tok != c:
                raise ValueError(
                    f"Token channel mismatch: got {c_tok}, expected {c} for target CNN shape"
                )

            side = int(round(math.sqrt(t)))
            if side * side == t:
                x = tokens.transpose(1, 2).reshape(bsz, c, side, side)
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
                return x

            x = tokens.transpose(1, 2)
            x = F.interpolate(x, size=h * w, mode="linear", align_corners=False)
            return x.reshape(bsz, c, h, w)

        if len(target_shape) == 2:
            n, c = target_shape
            t = tokens.size(1)
            c_tok = tokens.size(2)
            if c_tok != c:
                raise ValueError(
                    f"Token channel mismatch: got {c_tok}, expected {c} for target sequence shape"
                )
            if t != n:
                x = tokens.transpose(1, 2)
                x = F.interpolate(x, size=n, mode="linear", align_corners=False)
                tokens = x.transpose(1, 2)
            return tokens

        raise ValueError(f"Unsupported target shape template: {target_shape}")

    def _hidden_shape_wo_batch(self, x: torch.Tensor) -> Tuple[int, ...]:
        if x.dim() == 4:
            return (x.size(1), x.size(2), x.size(3))
        if x.dim() == 3:
            return (x.size(1), x.size(2))
        raise ValueError(f"Unsupported hidden shape: {tuple(x.shape)}")

    def _channel_dim(self, x: torch.Tensor) -> int:
        if x.dim() == 4:
            return x.size(1)
        if x.dim() == 3:
            return x.size(2)
        raise ValueError(f"Unsupported hidden shape: {tuple(x.shape)}")

    def _build_encoder_mlp(self, in_dim: int) -> nn.Module:
        hidden = max(self.z_dim, int(in_dim * self.mlp_expansion))
        hidden = min(hidden, 1024)
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.z_dim),
        )

    def _build_decoder_mlp(self, out_dim: int) -> nn.Module:
        hidden = max(self.z_dim, int(out_dim * self.mlp_expansion))
        hidden = min(hidden, 1024)
        return nn.Sequential(
            nn.Linear(self.z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def _ensure_layer_adapters(self, layer_name: str, hidden: torch.Tensor):
        key = self._layer_key(layer_name)
        ch = self._channel_dim(hidden)
        dev = hidden.device

        if key not in self.encoder_mlps:
            self.encoder_mlps[key] = self._build_encoder_mlp(ch).to(dev)
        if key not in self.decoder_mlps:
            self.decoder_mlps[key] = self._build_decoder_mlp(ch).to(dev)

        self.layer_input_channels[key] = ch
        self.layer_shape_templates[key] = self._hidden_shape_wo_batch(hidden)

    def _align_key(self, src_layer: str, tgt_layer: str) -> str:
        return f"{src_layer}__to__{tgt_layer}"

    def _ensure_transition_alignment(self, src_layer: str, tgt_layer: str, device=None):
        src_shape = self.layer_shape_templates[src_layer]
        tgt_shape = self.layer_shape_templates[tgt_layer]
        key = self._align_key(src_layer, tgt_layer)
        dev = device

        if len(src_shape) == 3 and len(tgt_shape) == 3:
            src_c = src_shape[0]
            tgt_c = tgt_shape[0]
            if src_c != tgt_c and key not in self.align_conv:
                self.align_conv[key] = nn.Conv2d(src_c, tgt_c, kernel_size=1, bias=False).to(dev)
            return

        if len(src_shape) == 2 and len(tgt_shape) == 2:
            src_c = src_shape[1]
            tgt_c = tgt_shape[1]
            if src_c != tgt_c and key not in self.align_linear:
                self.align_linear[key] = nn.Linear(src_c, tgt_c, bias=False).to(dev)
            return

        # Cross-modality or mixed channel format fallback.
        src_c = src_shape[0] if len(src_shape) == 3 else src_shape[1]
        tgt_c = tgt_shape[0] if len(tgt_shape) == 3 else tgt_shape[1]
        if src_c != tgt_c and key not in self.align_linear:
            self.align_linear[key] = nn.Linear(src_c, tgt_c, bias=False).to(dev)

    def ensure_from_features(self, features: Dict[str, torch.Tensor]):
        feature_dev = None
        for layer in self.layer_names:
            if layer not in features:
                raise KeyError(f"Layer '{layer}' missing in features: {list(features.keys())}")
            self._ensure_layer_adapters(layer, features[layer])
            if feature_dev is None:
                feature_dev = features[layer].device

        for src, tgt in zip(self.layer_names[:-1], self.layer_names[1:]):
            self._ensure_transition_alignment(src, tgt, device=feature_dev)

    def encode(self, hidden: torch.Tensor, layer_name: str) -> torch.Tensor:
        key = self._layer_key(layer_name)
        self._ensure_layer_adapters(key, hidden)

        tokens = self._tokenize(hidden)
        if self.use_token_norm:
            tokens = F.layer_norm(tokens, normalized_shape=(tokens.size(-1),))

        z = self.encoder_mlps[key](tokens)
        layer_bias = self.layer_embed.weight[self.layer_to_idx[key]].view(1, 1, -1)
        return z + layer_bias

    def decode(
        self,
        z: torch.Tensor,
        layer_name: str,
        target_shape_wo_batch: Tuple[int, ...],
    ) -> torch.Tensor:
        key = self._layer_key(layer_name)
        out_c = target_shape_wo_batch[0] if len(target_shape_wo_batch) == 3 else target_shape_wo_batch[1]

        if key not in self.decoder_mlps or self.layer_input_channels.get(key, None) != out_c:
            # Keep decoder keyed per layer but rebuild if channel schema changed.
            self.decoder_mlps[key] = self._build_decoder_mlp(out_c).to(z.device)
            self.layer_input_channels[key] = out_c

        layer_bias = self.layer_embed.weight[self.layer_to_idx[key]].view(1, 1, -1)
        tokens = self.decoder_mlps[key](z + layer_bias)
        return self._tokens_to_shape(tokens, target_shape_wo_batch)

    def encode_feature_dict(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.ensure_from_features(features)
        return {layer: self.encode(features[layer], layer) for layer in self.layer_names}

    def get_template_shape(self, layer_name: str, batch_size: int) -> Tuple[int, ...]:
        layer_name = self._layer_key(layer_name)
        if layer_name not in self.layer_shape_templates:
            raise KeyError(f"No shape template recorded for layer '{layer_name}'")
        return (batch_size, *self.layer_shape_templates[layer_name])

    def align_hidden(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        src_layer: str,
        tgt_layer: str,
    ) -> torch.Tensor:
        """
        Lightweight explicit alignment for transition deltas.

        - same shape: direct
        - CNN mismatch: resize spatial + optional 1x1 conv
        - sequence mismatch: resize token length + optional linear projection
        """
        if h_src.shape == h_tgt.shape:
            return h_src

        src_shape = self._hidden_shape_wo_batch(h_src)
        tgt_shape = self._hidden_shape_wo_batch(h_tgt)
        self.layer_shape_templates[src_layer] = src_shape
        self.layer_shape_templates[tgt_layer] = tgt_shape
        self._ensure_transition_alignment(src_layer, tgt_layer, device=h_src.device)
        key = self._align_key(src_layer, tgt_layer)

        if h_src.dim() == 4 and h_tgt.dim() == 4:
            x = h_src
            if x.shape[2:] != h_tgt.shape[2:]:
                x = F.interpolate(x, size=h_tgt.shape[2:], mode="bilinear", align_corners=False)
            if x.size(1) != h_tgt.size(1):
                x = self.align_conv[key](x)
            return x

        if h_src.dim() == 3 and h_tgt.dim() == 3:
            x = h_src
            if x.size(1) != h_tgt.size(1):
                x = x.transpose(1, 2)
                x = F.interpolate(x, size=h_tgt.size(1), mode="linear", align_corners=False)
                x = x.transpose(1, 2)
            if x.size(2) != h_tgt.size(2):
                x = self.align_linear[key](x)
            return x

        # Cross-modality fallback: map source to tokens then reconstruct to target shape.
        src_tokens = self._tokenize(h_src)
        tgt_shape_wo_batch = self._hidden_shape_wo_batch(h_tgt)

        target_tokens = tgt_shape_wo_batch[0] * tgt_shape_wo_batch[1] if len(tgt_shape_wo_batch) == 3 else tgt_shape_wo_batch[0]
        if src_tokens.size(1) != target_tokens:
            x = src_tokens.transpose(1, 2)
            x = F.interpolate(x, size=target_tokens, mode="linear", align_corners=False)
            src_tokens = x.transpose(1, 2)

        src_c = src_tokens.size(2)
        tgt_c = tgt_shape_wo_batch[0] if len(tgt_shape_wo_batch) == 3 else tgt_shape_wo_batch[1]
        if src_c != tgt_c:
            src_tokens = self.align_linear[key](src_tokens)

        return self._tokens_to_shape(src_tokens, tgt_shape_wo_batch)

    def compute_z_losses(
        self,
        features: Dict[str, torch.Tensor],
        lambda_rec: float = 1.0,
        lambda_dist: float = 1.0,
        lambda_angle: float = 1.0,
        lambda_trans_dist: float = 1.0,
        lambda_trans_angle: float = 1.0,
    ):
        self.ensure_from_features(features)

        z_by_layer: Dict[str, torch.Tensor] = OrderedDict()

        rec_terms = []
        dist_terms = []
        angle_terms = []
        trans_dist_terms = []
        trans_angle_terms = []

        details = {}

        for layer in self.layer_names:
            h = features[layer]
            z = self.encode(h, layer)
            z_by_layer[layer] = z

            h_hat = self.decode(z, layer, self.layer_shape_templates[layer])
            l_rec = F.mse_loss(h_hat, h)
            l_dist = rkd_distance_loss(z, h)
            l_angle = rkd_angle_loss(z, h)

            rec_terms.append(l_rec)
            dist_terms.append(l_dist)
            angle_terms.append(l_angle)

            details[f"rec_{layer}"] = l_rec.item()
            details[f"dist_{layer}"] = l_dist.item()
            details[f"angle_{layer}"] = l_angle.item()

        for src_layer, tgt_layer in zip(self.layer_names[:-1], self.layer_names[1:]):
            h_src = features[src_layer]
            h_tgt = features[tgt_layer]

            h_src_aligned = self.align_hidden(h_src, h_tgt, src_layer, tgt_layer)
            delta_h = h_tgt - h_src_aligned
            delta_z = z_by_layer[tgt_layer] - z_by_layer[src_layer]

            l_trans_dist = rkd_distance_loss(delta_z, delta_h)
            l_trans_angle = rkd_angle_loss(delta_z, delta_h)

            trans_dist_terms.append(l_trans_dist)
            trans_angle_terms.append(l_trans_angle)

            details[f"trans_dist_{src_layer}_to_{tgt_layer}"] = l_trans_dist.item()
            details[f"trans_angle_{src_layer}_to_{tgt_layer}"] = l_trans_angle.item()

        mean = lambda xs: torch.stack(xs).mean() if xs else torch.tensor(0.0, device=next(self.parameters()).device)

        rec = mean(rec_terms)
        dist = mean(dist_terms)
        angle = mean(angle_terms)
        trans_dist = mean(trans_dist_terms)
        trans_angle = mean(trans_angle_terms)

        total = (
            lambda_rec * rec
            + lambda_dist * dist
            + lambda_angle * angle
            + lambda_trans_dist * trans_dist
            + lambda_trans_angle * trans_angle
        )

        summary = {
            "total": total.item(),
            "rec": rec.item(),
            "dist": dist.item(),
            "angle": angle.item(),
            "trans_dist": trans_dist.item(),
            "trans_angle": trans_angle.item(),
        }
        summary.update(details)

        return total, summary, z_by_layer
