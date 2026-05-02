from typing import Dict, Tuple

import torch
import torch.nn as nn

from distill.flow_model import euler_integrate


RESNET_LAYER_ORDER = ["layer1", "layer2", "layer3", "layer4"]


class ResNetSegmentRunner(nn.Module):
    """Utility to run frozen torchvision ResNet in segments."""

    def __init__(self, teacher_wrapper: nn.Module):
        super().__init__()
        self.teacher_wrapper = teacher_wrapper
        self.model = teacher_wrapper.model

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        required = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
            "fc",
        ]
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise ValueError(
                "ResNetSegmentRunner requires torchvision ResNet-like model. "
                f"Missing attrs: {missing}"
            )

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        return x

    def forward_to_layer(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        if layer_name not in RESNET_LAYER_ORDER:
            raise ValueError(f"Unsupported layer '{layer_name}' for ResNet runner")

        x = self.stem(x)
        for name in RESNET_LAYER_ORDER:
            x = getattr(self.model, name)(x)
            if name == layer_name:
                return x

        raise RuntimeError(f"Could not reach layer '{layer_name}'")

    def forward_from_layer_output(self, layer_name: str, h: torch.Tensor) -> torch.Tensor:
        if layer_name not in RESNET_LAYER_ORDER:
            raise ValueError(f"Unsupported layer '{layer_name}' for ResNet runner")

        start_idx = RESNET_LAYER_ORDER.index(layer_name)
        x = h
        for name in RESNET_LAYER_ORDER[start_idx + 1 :]:
            x = getattr(self.model, name)(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ZeroShotFlowReplacement(nn.Module):
    """
    Zero-shot replacement wrapper using
      D_b ∘ EulerFlow_{a→b} ∘ E_a
    with frozen teacher stem/tail execution.
    """

    def __init__(
        self,
        teacher,
        zspace,
        flow,
        layer_names,
        start_layer: str,
        end_layer: str,
        num_steps: int = 4,
    ):
        super().__init__()
        self.teacher = teacher
        self.zspace = zspace
        self.flow = flow
        self.layer_names = list(layer_names)
        self.layer_to_idx = {name: i for i, name in enumerate(self.layer_names)}
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_steps = num_steps

        if start_layer not in self.layer_to_idx or end_layer not in self.layer_to_idx:
            raise ValueError(
                f"Unknown replacement layers: start={start_layer}, end={end_layer}, available={self.layer_names}"
            )

        self.runner = ResNetSegmentRunner(teacher)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def forward(self, x: torch.Tensor, return_details: bool = False):
        h_a = self.runner.forward_to_layer(x, self.start_layer)
        z_a = self.zspace.encode(h_a, self.start_layer)

        start_idx = self.layer_to_idx[self.start_layer]
        end_idx = self.layer_to_idx[self.end_layer]

        z_hat_b = euler_integrate(
            velocity_model=self.flow,
            z0=z_a,
            start_idx=start_idx,
            end_idx=end_idx,
            steps=self.num_steps,
        )

        if self.end_layer not in self.zspace.layer_shape_templates:
            raise KeyError(
                f"Missing shape template for layer '{self.end_layer}'. "
                "Run zspace.ensure_from_features(...) before zero-shot eval."
            )
        target_shape = self.zspace.layer_shape_templates[self.end_layer]
        h_hat_b = self.zspace.decode(z_hat_b, self.end_layer, target_shape)

        logits = self.runner.forward_from_layer_output(self.end_layer, h_hat_b)

        if return_details:
            return {
                "logits": logits,
                "z_a": z_a,
                "z_hat_b": z_hat_b,
                "h_a": h_a,
                "h_hat_b": h_hat_b,
            }

        return logits
