import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module, count_params


class FeatureHooks:
    """Helper class to extract intermediate features."""

    def __init__(self, named_layers):
        self.features = OrderedDict()
        self.hooks = []

        def hook_fn(name):
            def _hook(module, input, output):
                self.features[name] = output

            return _hook

        for name, layer in named_layers:
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))

    def clear(self):
        self.features.clear()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class FeatureRegressor(nn.Module):
    """1x1 Conv to project channels."""

    def __init__(self, in_channels, hidden_channels):
        super(FeatureRegressor, self).__init__()
        self.regressor = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        return self.regressor(x)


class FeatureDiscriminator(nn.Module):
    """Discriminator with Global Pooling. Returns logits."""

    def __init__(self, hidden_channels):
        super(FeatureDiscriminator, self).__init__()
        # Global pooling handles spatial mismatch automatically
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x):
        pooled = self.global_pool(x)
        output = self.discriminator(pooled)
        return output


class DisDKD(nn.Module):
    """
    DisDKD without MMD.
    Phase 1: Pure Adversarial Feature Alignment (GAN).
    Phase 2: Decoupled Knowledge Distillation (DKD).
    """

    def __init__(
        self,
        teacher,
        student,
        teacher_layer,
        student_layer,
        teacher_channels,
        student_channels,
        hidden_channels=256,
        alpha=1.0,
        beta=8.0,
        temperature=4.0,
        mmd_weight=0.0,  # Kept arg for compatibility, but ignored
    ):
        super(DisDKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        # mmd_weight is ignored in this version

        self.teacher_layer = teacher_layer
        self.student_layer = student_layer

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.teacher_hooks = FeatureHooks(
            [(teacher_layer, get_module(self.teacher.model, teacher_layer))]
        )
        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )

        # Regressors
        self.teacher_regressor = FeatureRegressor(teacher_channels, hidden_channels)
        # Freeze teacher regressor (Fixed Target)
        for p in self.teacher_regressor.parameters():
            p.requires_grad = False

        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)

        # Discriminator
        self.discriminator = FeatureDiscriminator(hidden_channels)
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.current_phase = 1
        self.adversarial_components_active = True

        print(
            f"Teacher regressor (Frozen): {count_params(self.teacher_regressor)*1e-6:.3f}M params"
        )
        print(
            f"Student regressor: {count_params(self.student_regressor)*1e-6:.3f}M params"
        )
        print(f"Discriminator: {count_params(self.discriminator)*1e-6:.3f}M params")
        print(f"DKD parameters: alpha={alpha}, beta={beta}, temp={temperature}")
        print("DisDKD Mode: Pure Adversarial Alignment (No MMD)")

    def set_phase(self, phase):
        assert phase in [1, 2]
        self.current_phase = phase

        if phase == 1:
            self.set_discriminator_mode()
            self.set_generator_mode()
            trainable = sum(
                p.numel() for p in self.student.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self.student.parameters())
            print(f"Phase 1: Student params active: {trainable:,}/{total:,}")
            self.set_discriminator_mode()

        elif phase == 2:
            self._unfreeze_student_completely()

    def _freeze_student_completely(self):
        for param in self.student.parameters():
            param.requires_grad = False

    def _unfreeze_student_completely(self):
        for param in self.student.parameters():
            param.requires_grad = True

    def _unfreeze_student_up_to_layer_g(self):
        self._freeze_student_completely()
        layer_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
        guided_layer_key = self.student_layer.split(".")[0]

        if guided_layer_key in layer_order:
            cutoff_idx = layer_order.index(guided_layer_key)
        else:
            cutoff_idx = len(layer_order) // 2

        layers_to_unfreeze = layer_order[: cutoff_idx + 1]
        for name, param in self.student.model.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name in layers_to_unfreeze:
                param.requires_grad = True

    def _freeze_student_regressor(self):
        for param in self.student_regressor.parameters():
            param.requires_grad = False

    def _unfreeze_student_regressor(self):
        for param in self.student_regressor.parameters():
            param.requires_grad = True

    def _freeze_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def _unfreeze_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def set_discriminator_mode(self):
        """Freeze Student/Regressor, Train Discriminator."""
        self._freeze_student_completely()
        self._freeze_student_regressor()
        self._unfreeze_discriminator()

    def set_generator_mode(self):
        """Freeze Discriminator, Train Student/Regressor."""
        self._freeze_discriminator()
        self._unfreeze_student_regressor()
        self._unfreeze_student_up_to_layer_g()

    def discard_adversarial_components(self):
        if not self.adversarial_components_active:
            return
        self.teacher_hooks.remove()
        self.student_hooks.remove()
        del self.teacher_regressor
        del self.student_regressor
        del self.discriminator
        self.teacher_regressor = None
        self.student_regressor = None
        self.discriminator = None
        self.adversarial_components_active = False
        print("Phase 2: Adversarial components discarded.")

    def compute_dkd_loss(self, logits_student, logits_teacher, target):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)

        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        # TCKD
        pred_student_tckd = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher_tckd = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student_tckd = torch.log(pred_student_tckd + 1e-8)

        tckd_loss = F.kl_div(
            log_pred_student_tckd, pred_teacher_tckd, reduction="batchmean"
        ) * (self.temperature**2)

        # NCKD
        pred_teacher_nckd = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_nckd = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )

        nckd_loss = F.kl_div(
            log_pred_student_nckd, pred_teacher_nckd, reduction="batchmean"
        ) * (self.temperature**2)

        return self.alpha * tckd_loss + self.beta * nckd_loss

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def discriminator_step(self, x):
        """Train discriminator to distinguish teacher (1) from student (0)."""
        batch_size = x.size(0)

        with torch.no_grad():
            _ = self.teacher(x)
            _ = self.student(x)

        teacher_feat = self.teacher_hooks.features.get(self.teacher_layer)
        student_feat = self.student_hooks.features.get(self.student_layer)

        # Project Features
        with torch.no_grad():
            teacher_hidden = self.teacher_regressor(teacher_feat)
            student_hidden = self.student_regressor(student_feat)
            # No spatial matching needed; global pool in discriminator handles it.

        # Predictions (Logits)
        teacher_logits = self.discriminator(teacher_hidden)
        student_logits = self.discriminator(
            student_hidden.detach()
        )  # Detach for D step

        # Label Smoothing (0.8 for Real, 0.2 for Fake) to stabilize GAN
        real_labels = torch.full((batch_size, 1), 0.8, device=x.device)
        fake_labels = torch.full((batch_size, 1), 0.2, device=x.device)

        loss_real = self.bce_loss(teacher_logits, real_labels)
        loss_fake = self.bce_loss(student_logits, fake_labels)
        disc_loss = (loss_real + loss_fake) / 2

        # Accuracy Calculation
        with torch.no_grad():
            t_acc = (torch.sigmoid(teacher_logits) > 0.5).float().sum()
            s_acc = (torch.sigmoid(student_logits) < 0.5).float().sum()
            disc_accuracy = (t_acc + s_acc) / (2 * batch_size)

        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return {
            "disc_loss": disc_loss,
            "disc_accuracy": disc_accuracy.item(),
        }

    def generator_step(self, x):
        """Train student to fool discriminator (output 1)."""
        batch_size = x.size(0)

        with torch.no_grad():
            _ = self.teacher(x)
        _ = self.student(x)  # Need gradients here

        teacher_feat = self.teacher_hooks.features.get(
            self.teacher_layer
        )  # Unused in loss, but needed to clear hooks?
        student_feat = self.student_hooks.features.get(self.student_layer)

        # Project
        student_hidden = self.student_regressor(student_feat)

        # Adversarial Loss: Student wants Disc output to be Real (1.0)
        student_logits = self.discriminator(student_hidden)
        real_labels = torch.ones(batch_size, 1, device=x.device)  # Hard 1.0 for fooling

        adv_loss = self.bce_loss(student_logits, real_labels)

        # Metrics
        with torch.no_grad():
            fool_rate = (torch.sigmoid(student_logits) > 0.5).float().mean()

        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return {
            "gen_loss": adv_loss,  # No MMD, just fooling loss
            "adv_loss": adv_loss.item(),
            "fool_rate": fool_rate.item(),
            "mmd": 0.0,  # Placeholder
        }

    def forward_phase2(self, x, targets):
        """Phase 2: DKD."""
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        dkd_loss = self.compute_dkd_loss(student_logits, teacher_logits, targets)

        return {"student_logits": student_logits, "dkd_loss": dkd_loss}

    def forward(self, x, targets=None, mode="discriminator"):
        if self.current_phase == 1:
            if mode == "discriminator":
                return self.discriminator_step(x)
            elif mode == "generator":
                return self.generator_step(x)
        elif self.current_phase == 2:
            return self.forward_phase2(x, targets)

    def get_discriminator_optimizer(self, lr=1e-4, weight_decay=1e-4):
        # Teacher regressor is frozen, so only train discriminator
        return torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, weight_decay=weight_decay
        )

    def get_generator_optimizer(self, lr=1e-4, weight_decay=1e-4):
        params = [p for p in self.student.parameters() if p.requires_grad] + list(
            self.student_regressor.parameters()
        )
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def get_dkd_optimizer(self, lr=1e-4, weight_decay=1e-4):
        return torch.optim.Adam(
            self.student.parameters(), lr=lr, weight_decay=weight_decay
        )
