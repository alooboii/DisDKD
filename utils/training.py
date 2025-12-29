import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils.utils import accuracy, AverageMeter
from utils.model_factory import create_distillation_model, print_model_parameters
from utils.checkpoint import save_checkpoint


class Trainer:
    """Handles model training and validation."""

    def __init__(
        self, teacher, student, num_classes, criterion, loss_tracker, device, args
    ):
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.criterion = criterion
        self.loss_tracker = loss_tracker
        self.device = device
        self.args = args

        # Create distillation model
        self.model = create_distillation_model(args, teacher, student, num_classes).to(
            device
        )
        print_model_parameters(self.model, args.method)

        # Setup optimizers
        self._setup_optimizers()

    def _setup_optimizers(self):
        """Setup optimizer(s) and scheduler(s)."""
        args = self.args

        # --- CONTRA-DKD LOGIC ---
        if args.method == "ContraDKD":
            self.student_optimizer, self.discriminator_optimizer = (
                self.model.get_optimizers(
                    student_lr=args.lr,
                    discriminator_lr=args.discriminator_lr,
                    weight_decay=args.weight_decay,
                )
            )
            self.student_scheduler = StepLR(
                self.student_optimizer, step_size=args.step_size, gamma=args.lr_decay
            )

        # --- DISDKD LOGIC (FIXED) ---
        elif args.method in ["DisDKD"]:
            # FIXED: Use DisDKD's get_optimizers() method which properly handles both optimizers
            self.student_optimizer, self.discriminator_optimizer = (
                self.model.get_optimizers(
                    student_lr=args.lr,
                    discriminator_lr=args.discriminator_lr,
                    weight_decay=args.weight_decay,
                )
            )
            self.student_scheduler = StepLR(
                self.student_optimizer, step_size=args.step_size, gamma=args.lr_decay
            )

        # --- STANDARD LOGIC ---
        else:
            self.student_optimizer, self.student_scheduler = self._create_optimizer(
                self.model
            )
            self.discriminator_optimizer = None

    def _create_optimizer(self, model):
        """Create optimizer and scheduler for a model."""
        args = self.args

        optimizers = {"adam": optim.Adam, "sgd": optim.SGD, "adamw": optim.AdamW}

        optimizer_class = optimizers[args.optimizer.lower()]
        optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}

        if args.optimizer.lower() == "sgd":
            optimizer_kwargs["momentum"] = args.momentum

        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

        return optimizer, scheduler

    def train(self, train_loader, val_loader):
        """Main training loop with FitNet 2-Stage Logic."""
        print(f"\nStarting training for {self.args.epochs} epochs...")

        if self.args.method == "ContraDKD":
            self.model.initialize_class_embeddings(train_loader, self.device)

        best_acc = 0.0

        # Store original weights for FitNet stage switching
        original_alpha = self.args.alpha
        original_beta = self.args.beta
        original_gamma = self.args.gamma

        for epoch in range(self.args.epochs):
            start_time = time.time()

            # --- FITNET STAGE SWITCHING LOGIC ---
            if self.args.method == "FitNet" and self.args.fitnet_stage1_epochs > 0:
                if epoch < self.args.fitnet_stage1_epochs:
                    self.args.alpha = 0.0
                    self.args.beta = 0.0
                    self.args.gamma = original_gamma
                    stage_name = "Stage 1 (Hint Only)"
                else:
                    self.args.alpha = original_alpha
                    self.args.beta = original_beta
                    self.args.gamma = 0.0
                    stage_name = "Stage 2 (Task)"
            else:
                stage_name = "Standard"

            # Train and validate
            if self.args.method in ["DisDKD", "ContraDKD"]:
                train_losses, train_acc = self._train_epoch_adversarial(
                    train_loader, epoch
                )
            else:
                train_losses, train_acc = self._train_epoch_standard(
                    train_loader, epoch
                )

            val_losses, val_acc = self._validate(val_loader, epoch)

            lr = self.student_optimizer.param_groups[0]["lr"]
            self.loss_tracker.log_epoch(epoch, "train", train_losses, train_acc, lr=lr)
            self.loss_tracker.log_epoch(epoch, "val", val_losses, val_acc, lr=lr)

            self.student_scheduler.step()

            # --- MERGED LOGGING LOGIC ---
            elapsed = time.time() - start_time

            if self.args.method == "DisDKD":
                disc_acc = train_losses.get("disc_accuracy", 0) * 100
                fool_rate = train_losses.get("fool_rate", 0) * 100
                dkd_loss = train_losses.get("dkd", 0)
                disc_loss = train_losses.get("discriminator", 0)
                adv_loss = train_losses.get("adversarial", 0)
                gp_loss = train_losses.get("gradient_penalty", 0)

                # Build log string
                log_str = (
                    f"Epoch {epoch}: Train {train_acc:.2f}%, Val {val_acc:.2f}% | "
                    f"Disc_Acc: {disc_acc:.1f}%, Fool: {fool_rate:.1f}% | "
                    f"DKD: {dkd_loss:.4f}, Disc: {disc_loss:.4f}, Adv: {adv_loss:.4f}"
                )
                if gp_loss > 0:
                    log_str += f", GP: {gp_loss:.4f}"
                log_str += f" | Time: {elapsed:.1f}s"
                print(log_str)

            elif self.args.method == "ContraDKD":
                disc_acc = train_losses.get("disc_accuracy", 0) * 100
                fool_rate = train_losses.get("fool_rate", 0) * 100
                dkd_loss = train_losses.get("dkd", 0)
                l2c_s = train_losses.get("l2c_student", 0)
                disc_loss = train_losses.get("discriminator", 0)
                adv_loss = train_losses.get("adversarial", 0)

                print(
                    f"Epoch {epoch}: Train {train_acc:.2f}%, Val {val_acc:.2f}% | "
                    f"D_Acc: {disc_acc:.1f}%, Fool: {fool_rate:.1f}% | "
                    f"DKD: {dkd_loss:.4f}, L2C(S): {l2c_s:.4f}, Disc: {disc_loss:.4f}, Adv: {adv_loss:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
            else:
                print(
                    f"Epoch {epoch} [{stage_name}]: Train {train_acc:.2f}%, Val {val_acc:.2f}%, Time {elapsed:.1f}s"
                )

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    self.model,
                    self.student_optimizer,
                    epoch,
                    val_acc,
                    self.args,
                    is_best=True,
                )

            print("-" * 80)

        return best_acc

    def _train_epoch_standard(self, train_loader, epoch):
        """Train for one epoch (standard methods)."""
        self.model.train()
        meters = self._init_meters()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            if self.args.method == "CRD":
                inputs, targets, indices = batch_data
                self.model.set_sample_indices(indices)
            else:
                inputs, targets = batch_data

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.student_optimizer.zero_grad()

            # Forward pass
            if self.args.method in ["DKD"]:
                teacher_logits, student_logits, method_specific_loss = self.model(
                    inputs, targets
                )
            else:
                teacher_logits, student_logits, method_specific_loss = self.model(
                    inputs
                )

            # Compute losses
            total_loss, losses_dict = self._compute_losses(
                teacher_logits, student_logits, targets, method_specific_loss, inputs
            )

            total_loss.backward()
            self.student_optimizer.step()

            # Update meters
            self._update_meters(
                meters, losses_dict, student_logits, targets, inputs.size(0)
            )

            # Update progress bar
            if batch_idx % max(1, self.args.print_freq // 10) == 0:
                self._update_progress_bar(progress_bar, meters)

        progress_bar.close()
        return self._get_average_losses(meters), meters["accuracy"].avg

    def _train_epoch_adversarial(self, train_loader, epoch):
        """Train for one epoch (adversarial methods: DisDKD, ContraDKD)."""
        self.model.train()
        meters = self._init_meters(adversarial=True)

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch} Train ({self.args.method})", leave=False
        )

        for batch_idx, batch_data in enumerate(progress_bar):
            inputs, targets = batch_data
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # IMPROVED: Adaptive discriminator training schedule
            train_disc = True

            if self.args.method == "ContraDKD":
                train_disc = batch_idx % 3 == 0  # Every 3rd batch
            elif self.args.method == "DisDKD":
                # Adaptive schedule for DisDKD
                if epoch < 3:
                    train_disc = batch_idx % 2 == 0  # More frequent early
                elif epoch < 10:
                    train_disc = batch_idx % 3 == 0  # Balanced mid
                else:
                    train_disc = batch_idx % 5 == 0  # Less frequent late

            if train_disc:
                # Step 1: Train Discriminator
                self.model.set_training_mode("discriminator")
                self.discriminator_optimizer.zero_grad()

                disc_result = self.model(inputs, targets)
                disc_loss = disc_result["total_disc_loss"]

                disc_loss.backward()
                self.discriminator_optimizer.step()
            else:
                # Just get metrics without training
                self.model.set_training_mode("discriminator")
                with torch.no_grad():
                    disc_result = self.model(inputs, targets)

            # Step 2: Train Student
            self.model.set_training_mode("student")
            self.student_optimizer.zero_grad()

            student_result = self.model(inputs, targets)
            teacher_logits = student_result["teacher_logits"]
            student_logits = student_result["student_logits"]

            # Compute standard losses
            ce_loss = self.criterion(student_logits, targets)

            # CRITICAL: Handle different loss structures
            if self.args.method == "DisDKD":
                # DisDKD separates DKD and adversarial losses
                # method_specific_loss = DKD only
                # total_student_loss = adversarial only
                kd_loss = torch.tensor(0.0, device=self.device)
                dkd_loss = student_result.get("method_specific_loss", 0)
                student_adv_loss = student_result["total_student_loss"]

                if isinstance(dkd_loss, torch.Tensor):
                    total_loss = (
                        self.args.alpha * ce_loss
                        + dkd_loss  # DKD loss
                        + self.args.disdkd_adversarial_weight
                        * student_adv_loss  # Adversarial
                    )
                else:
                    total_loss = (
                        self.args.alpha * ce_loss
                        + self.args.disdkd_adversarial_weight * student_adv_loss
                    )

            elif self.args.method == "ContraDKD":
                # ContraDKD combines everything in method_specific_loss
                # method_specific_loss = DKD + Adversarial + L2C
                kd_loss = torch.tensor(0.0, device=self.device)
                internal_loss = student_result["method_specific_loss"]
                total_loss = self.args.alpha * ce_loss + internal_loss

            else:
                # Fallback for any future adversarial methods
                kd_loss = self._compute_kd_loss(teacher_logits, student_logits)
                student_loss = student_result["total_student_loss"]
                total_loss = (
                    self.args.alpha * ce_loss
                    + self.args.beta * kd_loss
                    + self.args.gamma * student_loss
                )

            total_loss.backward()
            self.student_optimizer.step()

            # Update meters - combine metrics from both phases
            self._update_adversarial_meters(
                meters,
                disc_result,
                student_result,
                ce_loss,
                kd_loss,
                total_loss,
                student_logits,
                targets,
                inputs.size(0),
            )

            # Update progress bar
            if batch_idx % max(1, self.args.print_freq // 10) == 0:
                self._update_adversarial_progress_bar(progress_bar, meters)

        progress_bar.close()
        return self._get_average_losses(meters), meters["accuracy"].avg

    def _validate(self, val_loader, epoch=None):
        """Validate the model."""
        self.model.eval()

        if hasattr(self.model, "set_sample_indices"):
            self.model.set_sample_indices(None)

        meters = self._init_meters(adversarial=False)

        desc = f"Epoch {epoch} Val" if epoch is not None else "Validation"
        progress_bar = tqdm(val_loader, desc=desc, leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                if len(batch) == 3:
                    inputs, targets, indices = batch
                    indices = indices.to(self.device)
                else:
                    inputs, targets = batch
                    indices = None

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.args.method == "CRD" and indices is not None:
                    self.model.set_sample_indices(indices)

                # Handle ContraDKD same as DisDKD
                if self.args.method in ["DisDKD", "ContraDKD"]:
                    self.model.set_training_mode("student")
                    result = self.model(inputs, targets)
                    teacher_logits = result.get("teacher_logits", None)
                    student_logits = result["student_logits"]
                    method_specific_loss = result.get("method_specific_loss", None)
                elif self.args.method in ["DKD"]:
                    teacher_logits, student_logits, method_specific_loss = self.model(
                        inputs, targets
                    )
                else:
                    teacher_logits, student_logits, method_specific_loss = self.model(
                        inputs
                    )

                # Compute loss components
                ce_loss = self.criterion(student_logits, targets)
                kd_loss = self._compute_kd_loss(teacher_logits, student_logits)

                total = self.args.alpha * ce_loss + self.args.beta * kd_loss

                losses_dict = {
                    "ce": ce_loss.item(),
                    "kd": kd_loss.item(),
                }

                # Method-specific
                if method_specific_loss is not None and self.args.method in [
                    "CRD",
                    "FitNet",
                ]:
                    total = total + self.args.gamma * method_specific_loss
                    if self.args.method == "CRD":
                        losses_dict["contrastive"] = method_specific_loss.item()
                    elif self.args.method == "FitNet":
                        losses_dict["hint"] = method_specific_loss.item()

                losses_dict["total"] = total.item()

                # Update meters
                self._update_meters(
                    meters, losses_dict, student_logits, targets, inputs.size(0)
                )

                progress_bar.set_postfix(
                    {
                        "val_loss": f"{meters['total'].avg:.4f}",
                        "val_acc": f"{meters['accuracy'].avg:.2f}%",
                    }
                )

        progress_bar.close()
        return self._get_average_losses(meters), meters["accuracy"].avg

    def _compute_kd_loss(self, teacher_logits, student_logits):
        """Compute knowledge distillation loss."""
        if teacher_logits is None:
            return torch.tensor(0.0, device=student_logits.device)

        T = self.args.tau
        return nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(student_logits / T, dim=1),
            nn.functional.softmax(teacher_logits / T, dim=1),
        ) * (T * T)

    def _compute_losses(
        self, teacher_logits, student_logits, targets, method_specific_loss, inputs=None
    ):
        """Compute all losses for standard training."""
        ce_loss = self.criterion(student_logits, targets)

        if self.args.method in ["DKD"]:
            kd_loss = torch.tensor(0.0, device=student_logits.device)
        else:
            kd_loss = self._compute_kd_loss(teacher_logits, student_logits)

        weighted_ce = self.args.alpha * ce_loss
        weighted_kd = self.args.beta * kd_loss
        total_loss = weighted_ce + weighted_kd

        losses_dict = {
            "total": total_loss.item(),
            "ce": ce_loss.item(),
            "kd": kd_loss.item(),
        }

        # Handle method-specific losses
        if method_specific_loss is not None:
            if self.args.method in ["DKD"]:
                total_loss += method_specific_loss

                if self.args.method == "DKD":
                    tckd, nckd = self._compute_dkd_components(
                        student_logits, teacher_logits, targets
                    )
                    losses_dict["tckd"] = tckd
                    losses_dict["nckd"] = nckd

            else:
                method_loss_value = method_specific_loss.item()
                weighted_method = self.args.gamma * method_specific_loss
                total_loss += weighted_method

                loss_key_map = {
                    "FitNet": "hint",
                    "CRD": "contrastive",
                }
                loss_key = loss_key_map.get(self.args.method, "method_specific")
                losses_dict[loss_key] = method_loss_value

        losses_dict["total"] = total_loss.item()
        return total_loss, losses_dict

    def _compute_dkd_components(self, student_logits, teacher_logits, targets):
        """Compute TCKD and NCKD components for DKD logging."""
        with torch.no_grad():
            gt_mask = self.model._get_gt_mask(student_logits, targets)
            other_mask = self.model._get_other_mask(student_logits, targets)

            pred_student = torch.nn.functional.softmax(
                student_logits / self.args.tau, dim=1
            )
            pred_teacher = torch.nn.functional.softmax(
                teacher_logits / self.args.tau, dim=1
            )

            pred_student_tckd = self.model._cat_mask(pred_student, gt_mask, other_mask)
            pred_teacher_tckd = self.model._cat_mask(pred_teacher, gt_mask, other_mask)
            log_pred_student_tckd = torch.log(pred_student_tckd)

            tckd_loss = (
                torch.nn.functional.kl_div(
                    log_pred_student_tckd, pred_teacher_tckd, reduction="sum"
                )
                * (self.args.tau**2)
                / targets.shape[0]
            ).item()

            pred_teacher_nckd = torch.nn.functional.softmax(
                teacher_logits / self.args.tau - 1000.0 * gt_mask, dim=1
            )
            log_pred_student_nckd = torch.nn.functional.log_softmax(
                student_logits / self.args.tau - 1000.0 * gt_mask, dim=1
            )

            nckd_loss = (
                torch.nn.functional.kl_div(
                    log_pred_student_nckd, pred_teacher_nckd, reduction="sum"
                )
                * (self.args.tau**2)
                / targets.shape[0]
            ).item()

        return tckd_loss, nckd_loss

    def _init_meters(self, adversarial=False):
        """Initialize loss meters."""
        meters = {
            "total": AverageMeter(),
            "ce": AverageMeter(),
            "kd": AverageMeter(),
            "accuracy": AverageMeter(),
        }

        if adversarial:
            meters.update(
                {
                    "discriminator": AverageMeter(),
                    "adversarial": AverageMeter(),
                    "disc_accuracy": AverageMeter(),
                    "fool_rate": AverageMeter(),
                    "gradient_penalty": AverageMeter(),  # ADDED
                }
            )
            # Add method-specific meters for adversarial methods
            if self.args.method == "DisDKD":
                meters["dkd"] = AverageMeter()
            elif self.args.method == "ContraDKD":
                meters["dkd"] = AverageMeter()
                meters["l2c_student"] = AverageMeter()
        else:
            # Method-specific meters for standard training
            method_meters = {
                "FitNet": ["hint"],
                "CRD": ["contrastive"],
                "DKD": ["tckd", "nckd"],
            }

            if self.args.method in method_meters:
                for meter_name in method_meters[self.args.method]:
                    meters[meter_name] = AverageMeter()

        return meters

    def _update_meters(self, meters, losses_dict, student_logits, targets, batch_size):
        """Update meters with current batch losses."""
        acc1 = accuracy(student_logits, targets, topk=(1,))[0]

        for key, value in losses_dict.items():
            if key in meters:
                meters[key].update(value, batch_size)

        meters["accuracy"].update(acc1.item(), batch_size)

    def _update_adversarial_meters(
        self,
        meters,
        disc_result,
        student_result,
        ce_loss,
        kd_loss,
        total_loss,
        student_logits,
        targets,
        batch_size,
    ):
        """Update meters for adversarial training."""
        acc1 = accuracy(student_logits, targets, topk=(1,))[0]

        meters["total"].update(total_loss.item(), batch_size)
        meters["ce"].update(ce_loss.item(), batch_size)
        meters["kd"].update(kd_loss.item(), batch_size)
        meters["accuracy"].update(acc1.item(), batch_size)

        # Discriminator metrics
        meters["discriminator"].update(
            disc_result.get("discriminator_loss", 0), batch_size
        )
        meters["disc_accuracy"].update(
            disc_result.get("discriminator_accuracy", 0), batch_size
        )

        # ADDED: Gradient penalty tracking
        if "gradient_penalty" in disc_result:
            meters["gradient_penalty"].update(
                disc_result.get("gradient_penalty", 0), batch_size
            )

        # Student adversarial metrics
        meters["adversarial"].update(student_result.get("adversarial", 0), batch_size)
        meters["fool_rate"].update(student_result.get("fool_rate", 0), batch_size)

        # Method-specific metrics
        if self.args.method == "DisDKD":
            dkd_value = student_result.get("dkd", 0)
            if isinstance(dkd_value, torch.Tensor):
                dkd_value = dkd_value.item()
            meters["dkd"].update(dkd_value, batch_size)

        elif self.args.method == "ContraDKD":
            dkd_value = student_result.get("dkd", 0)
            l2c_value = student_result.get("l2c_student", 0)

            if isinstance(dkd_value, torch.Tensor):
                dkd_value = dkd_value.item()
            if isinstance(l2c_value, torch.Tensor):
                l2c_value = l2c_value.item()

            meters["dkd"].update(dkd_value, batch_size)
            meters["l2c_student"].update(l2c_value, batch_size)

    def _update_progress_bar(self, progress_bar, meters):
        """Update progress bar with current metrics."""
        postfix = {
            "loss": f'{meters["total"].avg:.4f}',
            "ce": f'{meters["ce"].avg:.4f}',
            "kd": f'{meters["kd"].avg:.4f}',
            "acc": f'{meters["accuracy"].avg:.2f}%',
        }

        if self.args.method == "DKD" and "tckd" in meters:
            postfix["tckd"] = f'{meters["tckd"].avg:.4f}'
            postfix["nckd"] = f'{meters["nckd"].avg:.4f}'

        progress_bar.set_postfix(postfix)

    def _update_adversarial_progress_bar(self, progress_bar, meters):
        """Update progress bar for adversarial methods."""
        postfix = {
            "loss": f'{meters["total"].avg:.4f}',
            "ce": f'{meters["ce"].avg:.4f}',
            "disc": f'{meters["discriminator"].avg:.4f}',
            "adv": f'{meters["adversarial"].avg:.4f}',
            "acc": f'{meters["accuracy"].avg:.2f}%',
            "disc_acc": f'{meters["disc_accuracy"].avg:.2%}',
            "fool": f'{meters["fool_rate"].avg:.2%}',
        }

        # ADDED: Show GP if available
        if "gradient_penalty" in meters and meters["gradient_penalty"].avg > 0:
            postfix["gp"] = f'{meters["gradient_penalty"].avg:.3f}'

        # Add method-specific progress info
        if self.args.method == "DisDKD" and "dkd" in meters:
            postfix["dkd"] = f'{meters["dkd"].avg:.4f}'
        elif self.args.method == "ContraDKD":
            if "dkd" in meters:
                postfix["dkd"] = f'{meters["dkd"].avg:.4f}'
            if "l2c_student" in meters:
                postfix["l2c"] = f'{meters["l2c_student"].avg:.4f}'

        progress_bar.set_postfix(postfix)

    def _get_average_losses(self, meters):
        """Return a dict of average losses/metrics (excluding accuracy)."""
        return {key: meter.avg for key, meter in meters.items() if key != "accuracy"}
