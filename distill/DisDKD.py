"""
DisDKD: Discriminator-guided Decoupled Knowledge Distillation

Training paradigm:
    Phase 1 (Adversarial - first N epochs, e.g., 10):
        Interleaved discriminator and generator training within each batch

        for batch in loader:
            # Train discriminator for k steps (typically k=1-2)
            for _ in range(k):
                model.set_discriminator_mode()
                outputs = model(x, mode='discriminator')
                loss = outputs['disc_loss']
                loss.backward()
                optimizer_D.step()

            # Train generator for 1 step
            model.set_generator_mode()
            outputs = model(x, mode='generator')
            loss = outputs['gen_loss']
            loss.backward()
            optimizer_G.step()

    Phase 2 (DKD - remaining epochs):
        Standard DKD training on entire student

        model.set_phase(2)
        model.discard_adversarial_components()

        for batch in loader:
            outputs = model(x, targets)
            loss = outputs['dkd_loss']
            loss.backward()
            optimizer_DKD.step()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module, count_params


class FeatureHooks:
    """
    Helper class to extract intermediate features from a network using forward hooks.
    """

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
        """Clears the stored features."""
        self.features.clear()

    def remove(self):
        """Removes all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class FeatureRegressor(nn.Module):
    """
    Light 1x1 convolutional regressor to project features to a common hidden dimension.
    """

    def __init__(self, in_channels, hidden_channels):
        super(FeatureRegressor, self).__init__()
        self.regressor = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        return self.regressor(x)


class FeatureDiscriminator(nn.Module):
    """
    Lightweight discriminator to distinguish between teacher and student features.
    Returns logits (not probabilities).
    """

    def __init__(self, hidden_channels):
        super(FeatureDiscriminator, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x):
        pooled = self.global_pool(x)
        output = self.discriminator(pooled)
        return output  # Returns logits


class DisDKD(nn.Module):
    """
    Discriminator-enhanced Decoupled Knowledge Distillation (Two-Phase with Interleaved Training).

    Phase 1 (Adversarial): Interleaved discriminator and generator training
        - Within each batch: Train D for k steps, then train G (student) for 1 step
        - Discriminator learns to distinguish teacher vs student features
        - Student (up to layer G) learns to fool discriminator
    Phase 2 (DKD): Pure DKD fine-tuning (entire student, regressors/discriminator discarded)

    Args:
        teacher (nn.Module): Pretrained teacher network
        student (nn.Module): Student network
        teacher_layer (str): Name of teacher layer for feature extraction
        student_layer (str): Name of student layer for feature extraction (layer G)
        teacher_channels (int): Number of channels in teacher feature map
        student_channels (int): Number of channels in student feature map
        hidden_channels (int): Number of channels in the common hidden space
        alpha (float): Weight for TCKD loss (DKD component)
        beta (float): Weight for NCKD loss (DKD component)
        temperature (float): Temperature for DKD softmax
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
        mmd_weight= 1.0,
    ):
        super(DisDKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.mmd_weight = mmd_weight
        self.teacher_layer = teacher_layer
        self.student_layer = student_layer

        # Freeze teacher parameters (always frozen)
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Set up hooks for feature extraction
        self.teacher_hooks = FeatureHooks(
            [(teacher_layer, get_module(self.teacher.model, teacher_layer))]
        )
        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )

        # Feature regressors to project to common dimension
        self.teacher_regressor = FeatureRegressor(teacher_channels, hidden_channels)
        for p in self.teacher_regressor.parameters():
            p.requires_grad = False
        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)

        # Feature discriminator
        self.discriminator = FeatureDiscriminator(hidden_channels)

        # BCE loss with logits for discriminator
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Track current phase (1=adversarial interleaved, 2=DKD)
        self.current_phase = 1

        # Track if adversarial components are active
        self.adversarial_components_active = True

        print(
            f"Teacher regressor: {count_params(self.teacher_regressor)*1e-6:.3f}M params"
        )
        print(
            f"Student regressor: {count_params(self.student_regressor)*1e-6:.3f}M params"
        )
        print(f"Discriminator: {count_params(self.discriminator)*1e-6:.3f}M params")
        print(f"DKD parameters: alpha={alpha}, beta={beta}, temperature={temperature}")

    def compute_mmd(self, x, y, sigma=1.0):
        """
        Computes Maximum Mean Discrepancy (MMD) with RBF kernel.
        x, y: [B, C]
        """
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        K_xx = torch.exp(-(rx.t() + rx - 2 * xx) / (2 * sigma ** 2))
        K_yy = torch.exp(-(ry.t() + ry - 2 * yy) / (2 * sigma ** 2))
        K_xy = torch.exp(-(rx.t() + ry - 2 * xy) / (2 * sigma ** 2))

        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd


    def set_phase(self, phase):
        """
        Set training phase and configure requires_grad accordingly.

        Phase 1: Adversarial (interleaved D/G training)
            - Discriminator and generator training alternate within each batch
            - Use set_discriminator_mode() and set_generator_mode() to switch
        Phase 2: DKD fine-tuning (entire student, adversarial components discarded)
        """
        assert phase in [1, 2], "Phase must be 1 (adversarial) or 2 (DKD)"
        self.current_phase = phase

        if phase == 1:
            # Phase 1: Adversarial - start in discriminator mode
            self.set_discriminator_mode()
            # Print student parameter info once
            self.set_generator_mode()  # Temporarily switch to see what gets trained
            trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.student.parameters())
            guided_layer_key = self.student_layer.split(".")[0]
            print(
                f"Phase 1: Student trainable params: {trainable:,} / {total:,} "
                f"({100*trainable/total:.1f}%) - up to {guided_layer_key}"
            )
            self.set_discriminator_mode()  # Switch back to discriminator mode

        elif phase == 2:
            # Phase 2: DKD - train entire student
            self._unfreeze_student_completely()

    def _freeze_student_completely(self):
        """Freeze all student parameters."""
        for param in self.student.parameters():
            param.requires_grad = False

    def _unfreeze_student_completely(self):
        """Unfreeze all student parameters."""
        for param in self.student.parameters():
            param.requires_grad = True

    def _unfreeze_student_up_to_layer_g(self):
        """
        Unfreeze student layers up to and including layer G (student_layer).
        Freeze layers after G (closer to output).

        For ResNet with student_layer='layer2':
        - Unfreeze: conv1, bn1, layer1, layer2
        - Freeze: layer3, layer4, fc
        """
        # First freeze everything
        self._freeze_student_completely()

        # Define layer order for ResNet
        layer_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]

        # Find the index of the guided layer
        guided_layer_key = self.student_layer.split(".")[0]
        if guided_layer_key in layer_order:
            cutoff_idx = layer_order.index(guided_layer_key)
        else:
            # Default: unfreeze first half
            cutoff_idx = len(layer_order) // 2
            print(
                f"Warning: {guided_layer_key} not in layer_order, using cutoff_idx={cutoff_idx}"
            )

        # Unfreeze layers up to and including the guided layer
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
        """
        Configure for discriminator training step in Phase 1.
        Freeze: student (entire), student_regressor
        Train: discriminator, teacher_regressor
        """
        self._freeze_student_completely()
        self._freeze_student_regressor()
        self._unfreeze_discriminator()

    def set_generator_mode(self):
        """
        Configure for generator (student) training step in Phase 1.
        Freeze: discriminator, teacher_regressor, student layers after G
        Train: student (up to layer G), student_regressor
        """
        self._freeze_discriminator()
        self._unfreeze_student_regressor()
        self._unfreeze_student_up_to_layer_g()

    def discard_adversarial_components(self):
        """
        Remove hooks and delete adversarial components for Phase 2 (DKD).
        Frees memory and ensures clean DKD training.
        """
        if not self.adversarial_components_active:
            return

        # Remove hooks
        self.teacher_hooks.remove()
        self.student_hooks.remove()

        # Delete components
        del self.teacher_regressor
        del self.student_regressor
        del self.discriminator

        self.teacher_regressor = None
        self.student_regressor = None
        self.discriminator = None
        self.adversarial_components_active = False

        print("Phase 2: Adversarial components discarded for DKD training")

    def match_spatial_dimensions(self, student_feat, teacher_feat):
        """Match spatial dimensions via interpolation."""
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.interpolate(
                student_feat,
                size=teacher_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return student_feat

    def compute_dkd_loss(self, logits_student, logits_teacher, target):
        """
        Compute the Decoupled Knowledge Distillation loss (TCKD + NCKD).
        """
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)

        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        # TCKD
        pred_student_tckd = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher_tckd = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student_tckd = torch.log(pred_student_tckd)

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
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def discriminator_step(self, x):
        """
        Discriminator training step for Phase 1 (adversarial).
        Train discriminator to distinguish teacher (1) from student (0).
        Call this with discriminator mode enabled (use set_discriminator_mode()).

        Returns dict with disc_loss and metrics.
        """
        batch_size = x.size(0)

        # Forward pass (student frozen, just need features)
        with torch.no_grad():
            _ = self.teacher(x)
            _ = self.student(x)

        # Extract features
        teacher_feat = self.teacher_hooks.features.get(self.teacher_layer)
        student_feat = self.student_hooks.features.get(self.student_layer)

        # Project to hidden space
        teacher_hidden = self.teacher_regressor(teacher_feat)
        with torch.no_grad():
            student_hidden = self.student_regressor(student_feat)

        # Match spatial dimensions
        student_hidden = self.match_spatial_dimensions(student_hidden, teacher_hidden)
        with torch.no_grad():
            t_pool = F.adaptive_avg_pool2d(teacher_hidden, 1).flatten(1)
            s_pool = F.adaptive_avg_pool2d(student_hidden, 1).flatten(1)
            t_pool = F.normalize(t_pool, dim=1)
            s_pool = F.normalize(s_pool, dim=1)
            mmd = self.compute_mmd(s_pool, t_pool)

        # Discriminator predictions (logits)
        teacher_logits = self.discriminator(teacher_hidden)
        student_logits = self.discriminator(student_hidden.detach())

        # Labels
        real_labels = torch.full((batch_size,1), 0.8, device=x.device)
        fake_labels = torch.full((batch_size,1), 0.2, device=x.device)

        # Discriminator loss (BCEWithLogitsLoss handles sigmoid internally)
        disc_loss_real = self.bce_loss(teacher_logits, real_labels)
        disc_loss_fake = self.bce_loss(student_logits, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2

        # Compute accuracy
        with torch.no_grad():
            teacher_pred = torch.sigmoid(teacher_logits)
            student_pred = torch.sigmoid(student_logits)
            teacher_correct = (teacher_pred > 0.5).float().sum()
            student_correct = (student_pred < 0.5).float().sum()
            disc_accuracy = (teacher_correct + student_correct) / (2 * batch_size)

        # Clear hooks
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return {
            "disc_loss": disc_loss,
            "disc_accuracy": disc_accuracy.item(),
            "teacher_pred_mean": teacher_pred.mean().item(),
            "student_pred_mean": student_pred.mean().item(),
            "mmd": mmd.item(),
        }

    def generator_step(self, x):
        batch_size = x.size(0)

        # Forward
        with torch.no_grad():
            _ = self.teacher(x)
        _ = self.student(x)

        # Extract features
        teacher_feat = self.teacher_hooks.features[self.teacher_layer]
        student_feat = self.student_hooks.features[self.student_layer]

        # Project
        with torch.no_grad():
            teacher_hidden = self.teacher_regressor(teacher_feat)
        student_hidden = self.student_regressor(student_feat)

        # Spatial match
        student_hidden = self.match_spatial_dimensions(student_hidden, teacher_hidden)

        # ---- MMD (GRADIENT FLOWS TO STUDENT) ----
        t_pool = F.adaptive_avg_pool2d(teacher_hidden, 1).flatten(1)
        s_pool = F.adaptive_avg_pool2d(student_hidden, 1).flatten(1)
        t_pool = F.normalize(t_pool, dim=1)
        s_pool = F.normalize(s_pool, dim=1)
        mmd = self.compute_mmd(s_pool, t_pool)

        # ---- Adversarial loss ----
        student_logits = self.discriminator(student_hidden)
        real_labels = torch.ones(batch_size, 1, device=x.device)
        adv_loss = self.bce_loss(student_logits, real_labels)

        # ---- FINAL GENERATOR LOSS ----
        gen_loss = adv_loss + self.mmd_weight * mmd

        # Metrics
        with torch.no_grad():
            student_pred = torch.sigmoid(student_logits)
            fool_rate = (student_pred > 0.5).float().mean()

        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return {
            "gen_loss": gen_loss,
            "adv_loss": adv_loss.item(),
            "mmd": mmd.item(),
            "fool_rate": fool_rate.item(),
        }


    def forward_phase2(self, x, targets):
        """
        Phase 2: Pure DKD training on entire student.
        Adversarial components should be discarded before this phase.
        """
        # Forward pass
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        # Compute DKD loss
        dkd_loss = self.compute_dkd_loss(student_logits, teacher_logits, targets)

        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "dkd_loss": dkd_loss,
        }

    def forward(self, x, targets=None, mode='discriminator'):
        """
        Forward pass that delegates to phase-specific methods.

        Args:
            x: Input tensor
            targets: Ground truth labels (required for Phase 2/DKD)
            mode: For Phase 1, specify 'discriminator' or 'generator' step
        """
        if self.current_phase == 1:
            # Phase 1: Adversarial (interleaved D/G)
            if mode == 'discriminator':
                return self.discriminator_step(x)
            elif mode == 'generator':
                return self.generator_step(x)
            else:
                raise ValueError(f"Invalid mode for Phase 1: {mode}. Use 'discriminator' or 'generator'")
        elif self.current_phase == 2:
            # Phase 2: DKD
            if targets is None:
                raise ValueError("Phase 2 requires targets for DKD loss")
            return self.forward_phase2(x, targets)
        else:
            raise ValueError(f"Invalid phase: {self.current_phase}")

    def get_discriminator_optimizer(self, lr=1e-4, weight_decay=1e-4):
        """
        Get optimizer for discriminator training in Phase 1 (adversarial).
        Trains: discriminator + teacher_regressor
        Default LR: 1e-4 (typical for discriminator training)
        """
        params = self.discriminator.parameters()
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def get_generator_optimizer(self, lr=1e-4, weight_decay=1e-4):
        """
        Get optimizer for generator (student) training in Phase 1 (adversarial).
        Trains: student (up to layer G) + student_regressor
        Default LR: 1e-4 or 5e-5 (typical for generator training)
        """
        params = [p for p in self.student.parameters() if p.requires_grad] + list(
            self.student_regressor.parameters()
        )
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def get_dkd_optimizer(self, lr=1e-4, weight_decay=1e-4):
        """
        Get optimizer for Phase 2 (DKD fine-tuning on entire student).
        Trains: entire student network
        """
        return torch.optim.Adam(
            self.student.parameters(), lr=lr, weight_decay=weight_decay
        )
