import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils import spectral_norm

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

        # Shallow spatial processing to keep receptive field without overfitting
        self.spatial = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shallow head to avoid overpowering the student during adversarial alignment.
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1),
            # No Sigmoid - returns logits
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.005)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.spatial(x)
        pooled_avg = self.global_pool(x)
        pooled_max = self.max_pool(x)
        pooled = torch.cat([pooled_avg, pooled_max], dim=1)
        output = self.discriminator(pooled)
        return output  # Returns logits


class DisDKD(nn.Module):
    """
    Discriminator-enhanced Decoupled Knowledge Distillation (Three-Phase).

    Phase 1: Pretrain discriminator and regressors (student backbone frozen)
    Phase 2: Adversarial feature alignment (student up to layer G, discriminator frozen)
    Phase 3: DKD fine-tuning (entire student, regressors/discriminator discarded)

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
        feature_noise_std=0.05,
        normalize_hidden=True,
        phase2_match_weight=0.0,
        adversarial_weight=1.0,
        gradient_penalty_weight=0.0,
    ):
        super(DisDKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.feature_noise_std = feature_noise_std
        self.normalize_hidden = normalize_hidden
        self.phase2_match_weight = phase2_match_weight
        self.adversarial_weight = adversarial_weight
        self.gradient_penalty_weight = gradient_penalty_weight

        self.teacher_layer = teacher_layer
        self.student_layer = student_layer
        self.phase2_layers_to_train = None

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
        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)

        # Feature discriminator
        self.discriminator = FeatureDiscriminator(hidden_channels)

        # BCE loss with logits for discriminator
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Track current phase (1, 2, or 3)
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
        print(
            f"Feature preprocessing: noise std={feature_noise_std}, "
            f"standardization={'on' if normalize_hidden else 'off'}, "
            f"phase2_match_weight={phase2_match_weight}, "
            f"adversarial_weight={adversarial_weight}, "
            f"gradient_penalty={gradient_penalty_weight}"
        )

    def set_phase(self, phase):
        """
        Set training phase and configure requires_grad accordingly.

        Phase 1: Train discriminator + both regressors (student backbone frozen)
        Phase 2: Train student (up to layer G) + student_regressor (discriminator frozen)
        Phase 3: Train entire student with DKD (adversarial components discarded)
        """
        assert phase in [1, 2, 3], "Phase must be 1, 2, or 3"
        self.current_phase = phase

        if phase == 1:
            # Freeze: entire student backbone
            # Train: discriminator, teacher_regressor, student_regressor
            self.phase2_layers_to_train = None
            self._freeze_student_completely()
            self._unfreeze_student_regressor()
            self._unfreeze_discriminator()
            self._unfreeze_teacher_regressor()

            # Ensure dropout/batchnorm behave correctly while training discriminator
            self.discriminator.train()
            self.teacher_regressor.train()
            self.student_regressor.train()
            self.teacher.eval()
            self.student.eval()

        elif phase == 2:
            # Freeze: discriminator, teacher_regressor, student layers after G
            # Train: student (up to G), student_regressor
            self._freeze_discriminator()
            self._freeze_teacher_regressor()
            self._unfreeze_student_regressor()
            layers_to_train = self._unfreeze_student_up_to_layer_g()
            self.phase2_layers_to_train = layers_to_train

            # Keep frozen modules in eval mode so dropout does not corrupt logits
            self.discriminator.eval()
            self.teacher_regressor.eval()
            self.student_regressor.train()
            self.teacher.eval()
            self._set_student_train_mode_up_to_layer_g(layers_to_train)

        elif phase == 3:
            # Train: entire student
            # Discard: regressors and discriminator (handled separately)
            self._unfreeze_student_completely()
            self.teacher.eval()
            self.student.train()

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

        # Count trainable params for logging
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student.parameters())
        print(
            f"Phase 2: Student trainable params: {trainable:,} / {total:,} "
            f"({100*trainable/total:.1f}%) - up to {guided_layer_key}"
        )

        return layers_to_unfreeze

    def _set_student_train_mode_up_to_layer_g(self, layers_to_unfreeze):
        """Put only layers up to G in train mode (recursively) and keep later layers in eval mode."""
        if not hasattr(self.student, "model"):
            # Fallback: default to train if model wrapper is missing
            self.student.train()
            return

        def set_mode_recursive(module, parent_name, should_train):
            if should_train:
                module.train()
            else:
                module.eval()

            for child_name, child in module.named_children():
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                child_root = full_name.split(".")[0]
                child_should_train = child_root in layers_to_unfreeze
                set_mode_recursive(child, full_name, child_should_train)

        for name, module in self.student.model.named_children():
            should_train = name in layers_to_unfreeze
            set_mode_recursive(module, name, should_train)

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

    def _freeze_teacher_regressor(self):
        for param in self.teacher_regressor.parameters():
            param.requires_grad = False

    def _unfreeze_teacher_regressor(self):
        for param in self.teacher_regressor.parameters():
            param.requires_grad = True

    def discard_adversarial_components(self):
        """
        Remove hooks and delete adversarial components for Phase 3.
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

        print("Phase 3: Adversarial components discarded")

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

    def _batch_normalize_pair(self, teacher_hidden, student_hidden):
        """Apply shared batch/channel normalization across teacher and student features."""
        combined = torch.cat([teacher_hidden, student_hidden], dim=0)
        mean = combined.mean(dim=(0, 2, 3), keepdim=True)
        std = combined.var(dim=(0, 2, 3), keepdim=True, unbiased=False).sqrt() + 1e-6
        normalized = (combined - mean) / std
        teacher_norm, student_norm = normalized.chunk(2, dim=0)
        return teacher_norm, student_norm

    def _normalize_hidden(self, hidden):
        if not self.normalize_hidden:
            return hidden

        # Instance normalization: per-sample, per-channel over spatial dimensions
        mean = hidden.mean(dim=(2, 3), keepdim=True)
        std = hidden.var(dim=(2, 3), keepdim=True, unbiased=False).sqrt() + 1e-6
        return (hidden - mean) / std

    def _preprocess_hidden(self, hidden, add_noise=False):
        if (
            add_noise
            and self.training
            and self.feature_noise_std > 0
        ):
            hidden = hidden + torch.randn_like(hidden) * self.feature_noise_std
        return self._normalize_hidden(hidden)

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

    def forward_phase1(self, x):
        """
        Phase 1: Train discriminator to distinguish teacher (1) from student (0).
        Student backbone is frozen but regressors are trainable.
        Discriminator outputs logits, not probabilities.
        """
        try:
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
            student_hidden = self.student_regressor(student_feat)

            # Match spatial dimensions before preprocessing
            student_hidden = self.match_spatial_dimensions(student_hidden, teacher_hidden)

            # Normalize / add noise so discriminator cannot rely on scale shortcuts
            teacher_hidden = self._preprocess_hidden(teacher_hidden, add_noise=True)
            student_hidden = self._preprocess_hidden(student_hidden)
            teacher_hidden, student_hidden = self._batch_normalize_pair(
                teacher_hidden, student_hidden
            )

            # Discriminator predictions (logits)
            teacher_logits = self.discriminator(teacher_hidden)
            student_logits = self.discriminator(student_hidden)

            # Hard labels to encourage a strong decision boundary
            real_labels = torch.ones(batch_size, 1, device=x.device)
            fake_labels = torch.zeros(batch_size, 1, device=x.device)

            # Discriminator loss (BCEWithLogitsLoss handles sigmoid internally)
            disc_loss_real = self.bce_loss(teacher_logits, real_labels)
            disc_loss_fake = self.bce_loss(student_logits, fake_labels)
            disc_loss = (disc_loss_real + disc_loss_fake) / 2

            # Optional gradient penalty to stabilize discriminator training
            if (
                self.training
                and hasattr(self, "gradient_penalty_weight")
                and self.gradient_penalty_weight > 0
            ):
                alpha = torch.rand(batch_size, 1, 1, 1, device=x.device)
                interpolated = (
                    alpha * teacher_hidden + (1 - alpha) * student_hidden
                ).requires_grad_(True)

                disc_interpolated = self.discriminator(interpolated)
                gradients = torch.autograd.grad(
                    outputs=disc_interpolated,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(disc_interpolated),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                gradients = gradients.view(batch_size, -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                disc_loss = disc_loss + self.gradient_penalty_weight * gradient_penalty

            # Compute accuracy for monitoring only (no adaptive scaling)
            with torch.no_grad():
                all_logits = torch.cat([teacher_logits, student_logits], dim=0)
                all_labels = torch.cat([real_labels, fake_labels], dim=0)
                all_preds = (torch.sigmoid(all_logits) > 0.5).float()
                disc_accuracy = (all_preds == all_labels).float().mean()

                teacher_pred = torch.sigmoid(teacher_logits)
                student_pred = torch.sigmoid(student_logits)

            disc_loss = disc_loss * loss_scale

            return {
                "disc_loss": disc_loss,
                "disc_accuracy": disc_accuracy.item(),
                "teacher_pred_mean": teacher_pred.mean().item(),
                "student_pred_mean": student_pred.mean().item(),
            }
        finally:
            # Ensure hooks are always cleared
            self.teacher_hooks.clear()
            self.student_hooks.clear()

    def forward_phase2(self, x):
        """
        Phase 2: Train student (up to layer G) to fool frozen discriminator.
        Pure adversarial, no CE loss (frozen fc produces meaningless logits).
        Discriminator outputs logits, not probabilities.
        """
        try:
            batch_size = x.size(0)

            # Forward pass
            with torch.no_grad():
                _ = self.teacher(x)
            _ = self.student(x)  # Student forward (partial grad flow)

            # Extract features
            teacher_feat = self.teacher_hooks.features.get(self.teacher_layer)
            student_feat = self.student_hooks.features.get(self.student_layer)

            # Project to hidden space
            with torch.no_grad():
                teacher_hidden = self.teacher_regressor(teacher_feat)
            student_hidden = self.student_regressor(student_feat)

            # Match spatial dimensions
            student_hidden = self.match_spatial_dimensions(student_hidden, teacher_hidden)

            # Normalize / add noise so discriminator cannot rely on scale shortcuts
            teacher_hidden = self._preprocess_hidden(teacher_hidden, add_noise=True)
            student_hidden = self._preprocess_hidden(student_hidden)

            # Apply shared batch normalization for fair feature matching
            teacher_hidden, student_hidden = self._batch_normalize_pair(
                teacher_hidden, student_hidden
            )

            # Adversarial loss: student wants to be classified as teacher (1)
            student_logits = self.discriminator(student_hidden)
            real_labels = torch.ones(batch_size, 1, device=x.device)
            adversarial_loss = self.bce_loss(student_logits, real_labels)

            feature_match_loss = torch.tensor(0.0, device=x.device)
            if self.phase2_match_weight > 0:
                feature_match_loss = (
                    F.mse_loss(student_hidden, teacher_hidden, reduction="sum")
                    / batch_size
                )

            total_loss = (
                self.adversarial_weight * adversarial_loss
                + self.phase2_match_weight * feature_match_loss
                - self.diversity_weight * diversity_loss
            )

            # Compute fool rate for early stopping check
            with torch.no_grad():
                student_pred = torch.sigmoid(student_logits)  # Convert to probabilities
                fool_rate = (student_pred > 0.5).float().mean()

            return {
                "adversarial_loss": adversarial_loss,
                "feature_match_loss": feature_match_loss,
                "diversity_loss": diversity_loss,
                "total_loss": total_loss,
                "fool_rate": fool_rate.item(),
                "student_pred_mean": student_pred.mean().item(),
            }
        finally:
            # Ensure hooks are always cleared
            self.teacher_hooks.clear()
            self.student_hooks.clear()

    def forward_phase3(self, x, targets):
        """
        Phase 3: Pure DKD training on entire student.
        Adversarial components should be discarded before this.
        """
        try:
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
        finally:
            self.teacher_hooks.clear()
            self.student_hooks.clear()

    def forward(self, x, targets=None):
        """
        Forward pass that delegates to phase-specific methods.
        """
        if self.current_phase == 1:
            return self.forward_phase1(x)
        elif self.current_phase == 2:
            return self.forward_phase2(x)
        elif self.current_phase == 3:
            if targets is None:
                raise ValueError("Phase 3 requires targets for DKD loss")
            return self.forward_phase3(x, targets)
        else:
            raise ValueError(f"Invalid phase: {self.current_phase}")

    def get_phase1_optimizer(self, lr=1e-3, weight_decay=1e-4):
        """Get optimizer for Phase 1 (discriminator + both regressors)."""
        params = (
            list(self.discriminator.parameters())
            + list(self.teacher_regressor.parameters())
            + list(self.student_regressor.parameters())
        )
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def get_phase2_optimizer(self, lr=1e-3, weight_decay=1e-4):
        """Get optimizer for Phase 2 (student up to G + student_regressor)."""
        params = [p for p in self.student.parameters() if p.requires_grad] + list(
            self.student_regressor.parameters()
        )
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def get_phase3_optimizer(self, lr=1e-4, weight_decay=1e-4):
        """Get optimizer for Phase 3 (entire student)."""
        return torch.optim.Adam(
            self.student.parameters(), lr=lr, weight_decay=weight_decay
        )
