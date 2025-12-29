import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module, count_params


class FeatureHooks:
    """Helper class to extract intermediate features using forward hooks."""

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
    1x1 convolutional regressor with BatchNorm for stable feature projection.

    Args:
        in_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels for common space
    """

    def __init__(self, in_channels, hidden_channels):
        super(FeatureRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),  # ADDED: Stabilizes feature statistics
        )

    def forward(self, x):
        return self.regressor(x)


class FeatureDiscriminator(nn.Module):
    """
    Enhanced discriminator with spectral normalization and better activations.

    Args:
        hidden_channels (int): Number of channels in the hidden feature space
        use_spectral_norm (bool): Whether to use spectral normalization
    """

    def __init__(self, hidden_channels, use_spectral_norm=True):
        super(FeatureDiscriminator, self).__init__()

        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Helper function to apply spectral normalization
        def maybe_sn(layer):
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        # Enhanced discriminator with spectral norm and LeakyReLU
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            maybe_sn(nn.Linear(hidden_channels, hidden_channels // 2)),
            nn.LeakyReLU(0.2, inplace=True),  # CHANGED: Better for adversarial training
            nn.Dropout(0.3),  # INCREASED: More regularization
            maybe_sn(nn.Linear(hidden_channels // 2, hidden_channels // 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            maybe_sn(nn.Linear(hidden_channels // 4, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features [batch_size, hidden_channels, H, W]

        Returns:
            Tensor: Discriminator output [batch_size, 1]
        """
        pooled = self.global_pool(x)
        output = self.discriminator(pooled)
        return output


class DisDKD(nn.Module):
    """
    Improved Discriminator-enhanced Decoupled Knowledge Distillation.

    Enhancements:
    - BatchNorm in feature regressors
    - Spectral normalization in discriminator
    - Label smoothing for stable training
    - Gradient penalty for improved stability
    - Feature normalization
    - Adaptive discriminator training

    Args:
        teacher (nn.Module): Pretrained teacher network
        student (nn.Module): Student network
        teacher_layer (str): Name of teacher layer for feature extraction
        student_layer (str): Name of student layer for feature extraction
        teacher_channels (int): Number of channels in teacher feature map
        student_channels (int): Number of channels in student feature map
        hidden_channels (int): Number of channels in the common hidden space
        alpha (float): Weight for TCKD loss (DKD component)
        beta (float): Weight for NCKD loss (DKD component)
        temperature (float): Temperature for DKD softmax
        use_spectral_norm (bool): Whether to use spectral normalization
        use_gradient_penalty (bool): Whether to use gradient penalty
        label_smoothing (float): Label smoothing factor (0.0 to 0.5)
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
        use_spectral_norm=True,
        use_gradient_penalty=True,
        label_smoothing=0.1,
        gp_lambda=5.0,
    ):
        super(DisDKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        # ADDED: Stabilization parameters
        self.use_gradient_penalty = use_gradient_penalty
        self.gp_lambda = gp_lambda
        self.label_smoothing = label_smoothing

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Set up hooks for feature extraction
        self.teacher_hooks = FeatureHooks(
            [(teacher_layer, get_module(self.teacher.model, teacher_layer))]
        )
        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )

        self.teacher_layer = teacher_layer
        self.student_layer = student_layer

        # IMPROVED: Feature regressors with BatchNorm
        self.teacher_regressor = FeatureRegressor(teacher_channels, hidden_channels)
        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)

        # IMPROVED: Discriminator with spectral normalization
        self.discriminator = FeatureDiscriminator(hidden_channels, use_spectral_norm)

        # BCE loss for discriminator
        self.bce_loss = nn.BCELoss()

        # Track training mode
        self.training_mode = "student"

        print(
            f"Teacher regressor: {count_params(self.teacher_regressor)*1e-6:.3f}M params"
        )
        print(
            f"Student regressor: {count_params(self.student_regressor)*1e-6:.3f}M params"
        )
        print(f"Discriminator: {count_params(self.discriminator)*1e-6:.3f}M params")
        print(f"DKD params: α={alpha}, β={beta}, T={temperature}")
        print(
            f"Spectral Norm: {use_spectral_norm}, GP: {use_gradient_penalty} (λ={gp_lambda})"
        )
        print(f"Label Smoothing: {label_smoothing}\n")

    def set_training_mode(self, mode):
        """Set training mode: 'student' or 'discriminator'"""
        assert mode in ["student", "discriminator"], "Invalid mode"
        self.training_mode = mode

        if mode == "discriminator":
            # Freeze student, unfreeze discriminator + teacher projector
            for param in self.student.parameters():
                param.requires_grad = False
            for param in self.student_regressor.parameters():
                param.requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = True
            for param in self.teacher_regressor.parameters():
                param.requires_grad = True
        else:  # student mode
            # Unfreeze student, freeze discriminator + teacher projector
            for param in self.student.parameters():
                param.requires_grad = True
            for param in self.student_regressor.parameters():
                param.requires_grad = True
            for param in self.discriminator.parameters():
                param.requires_grad = False
            for param in self.teacher_regressor.parameters():
                param.requires_grad = False

    def compute_gradient_penalty(self, real_features, fake_features):
        """
        Compute WGAN-GP style gradient penalty for discriminator stability.

        Args:
            real_features: Teacher features [B, C, H, W]
            fake_features: Student features [B, C, H, W]

        Returns:
            Gradient penalty term
        """
        batch_size = real_features.size(0)

        # ADD THIS BLOCK
        if real_features.shape[2:] != fake_features.shape[2:]:
            real_features = F.adaptive_avg_pool2d(real_features, 1)
            fake_features = F.adaptive_avg_pool2d(fake_features, 1)

        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_features.device)

        # Create interpolated features
        interpolates = alpha * real_features + (1 - alpha) * fake_features
        interpolates.requires_grad_(True)

        # Get discriminator output for interpolated features
        disc_interpolates = self.discriminator(interpolates)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Flatten and compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def compute_dkd_loss(self, logits_student, logits_teacher, target):
        """
        Compute Decoupled Knowledge Distillation loss (TCKD + NCKD).

        Args:
            logits_student: Student logits
            logits_teacher: Teacher logits
            target: Ground truth labels

        Returns:
            Combined DKD loss
        """
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)

        # Softmax with temperature
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        # TCKD: Target Class Knowledge Distillation
        pred_student_tckd = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher_tckd = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student_tckd = torch.log(pred_student_tckd + 1e-8)  # Add epsilon

        tckd_loss = F.kl_div(
            log_pred_student_tckd, pred_teacher_tckd, reduction="batchmean"
        ) * (self.temperature**2)

        # NCKD: Non-Target Class Knowledge Distillation
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
        """Create mask for ground truth class."""
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()

    def _get_other_mask(self, logits, target):
        """Create mask for non-ground truth classes."""
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()

    def _cat_mask(self, t, mask1, mask2):
        """Concatenate masked probabilities."""
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def forward(self, x, targets):
        """
        Forward pass with improved feature processing.

        Args:
            x: Input tensor
            targets: Ground truth labels

        Returns:
            Dictionary containing logits and losses
        """
        batch_size = x.size(0)

        # Forward pass
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        # Extract features
        teacher_feat = self.teacher_hooks.features.get(self.teacher_layer)
        student_feat = self.student_hooks.features.get(self.student_layer)

        if teacher_feat is None or student_feat is None:
            raise ValueError(
                f"Missing features for {self.teacher_layer} or {self.student_layer}"
            )

        # Project features to common dimension
        teacher_hidden = self.teacher_regressor(teacher_feat)
        student_hidden = self.student_regressor(student_feat)

        # ADDED: Normalize features for more stable training
        teacher_hidden = F.normalize(teacher_hidden, dim=1)
        student_hidden = F.normalize(student_hidden, dim=1)

        result = {"teacher_logits": teacher_logits, "student_logits": student_logits}

        if self.training_mode == "discriminator":
            # ========== DISCRIMINATOR TRAINING PHASE ==========

            # Get discriminator predictions
            teacher_pred = self.discriminator(teacher_hidden)
            student_pred = self.discriminator(student_hidden.detach())

            # IMPROVED: Label smoothing for stability
            real_labels = torch.ones(batch_size, 1, device=x.device) * (
                1.0 - self.label_smoothing
            )
            fake_labels = (
                torch.ones(batch_size, 1, device=x.device) * self.label_smoothing
            )

            # Discriminator loss
            disc_loss_real = self.bce_loss(teacher_pred, real_labels)
            disc_loss_fake = self.bce_loss(student_pred, fake_labels)
            disc_loss = (disc_loss_real + disc_loss_fake) / 2

            # ADDED: Gradient penalty for stability
            if self.use_gradient_penalty and torch.is_grad_enabled():
                gp = self.compute_gradient_penalty(teacher_hidden.detach(), student_hidden.detach())
                disc_loss = disc_loss + self.gp_lambda * gp
                result["gradient_penalty"] = gp.item()

            # Calculate discriminator accuracy
            teacher_correct = (teacher_pred > 0.5).float()
            student_correct = (student_pred <= 0.5).float()
            disc_accuracy = (
                (teacher_correct.sum() + student_correct.sum()) / (2 * batch_size)
            ).item()

            result["discriminator_loss"] = disc_loss.item()
            result["discriminator_accuracy"] = disc_accuracy
            result["total_disc_loss"] = disc_loss

        else:  # student mode
            # ========== STUDENT TRAINING PHASE ==========

            # DKD loss (logit-level distillation)
            dkd_loss = self.compute_dkd_loss(student_logits, teacher_logits, targets)

            # Adversarial loss (feature-level alignment)
            student_pred = self.discriminator(student_hidden)
            real_labels = torch.ones(batch_size, 1, device=x.device)
            adversarial_loss = self.bce_loss(student_pred, real_labels)

            # Calculate fool rate
            fool_rate = (student_pred > 0.5).float().mean().item()

            result["dkd"] = dkd_loss.item()
            result["adversarial"] = adversarial_loss.item()
            result["fool_rate"] = fool_rate
            result["total_student_loss"] = adversarial_loss
            result["method_specific_loss"] = dkd_loss

        # Clear features
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return result

    def get_optimizers(self, student_lr=1e-3, discriminator_lr=1e-4, weight_decay=1e-4):
        """
        Get separate optimizers for student and discriminator.

        Args:
            student_lr: Learning rate for student
            discriminator_lr: Learning rate for discriminator
            weight_decay: Weight decay

        Returns:
            (student_optimizer, discriminator_optimizer)
        """
        student_params = list(self.student.parameters()) + list(
            self.student_regressor.parameters()
        )

        discriminator_params = list(self.discriminator.parameters()) + list(
            self.teacher_regressor.parameters()
        )

        # Use AdamW for better regularization
        student_optimizer = torch.optim.AdamW(
            student_params, lr=student_lr, weight_decay=weight_decay
        )

        discriminator_optimizer = torch.optim.AdamW(
            discriminator_params, lr=discriminator_lr, weight_decay=weight_decay
        )

        return student_optimizer, discriminator_optimizer
