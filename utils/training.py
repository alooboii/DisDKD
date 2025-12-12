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

        # Setup optimizers (DisDKD handles its own optimizers per phase)
        if args.method != "DisDKD":
            self._setup_optimizers()

    def _setup_optimizers(self):
        """Setup optimizer(s) and scheduler(s) for non-DisDKD methods."""
        args = self.args
        self.student_optimizer, self.student_scheduler = self._create_optimizer(
            self.model
        )
        self.discriminator_optimizer = None

    def _create_optimizer(self, model, lr=None):
        """Create optimizer and scheduler for a model."""
        args = self.args
        lr = lr or args.lr

        optimizers = {"adam": optim.Adam, "sgd": optim.SGD, "adamw": optim.AdamW}

        optimizer_class = optimizers[args.optimizer.lower()]
        optimizer_kwargs = {"lr": lr, "weight_decay": args.weight_decay}

        if args.optimizer.lower() == "sgd":
            optimizer_kwargs["momentum"] = args.momentum

        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

        return optimizer, scheduler

    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"\nStarting training for {self.args.epochs} epochs...")
        best_acc = 0.0

        if self.args.method == "DisDKD":
            return self._train_disdkd(train_loader, val_loader)

        for epoch in range(self.args.epochs):
            start_time = time.time()

            # Train and validate
            train_losses, train_acc = self._train_epoch_standard(train_loader, epoch)
            val_losses, val_acc = self._validate(val_loader, epoch)

            # Log losses
            self.loss_tracker.log_epoch(epoch, "train", train_losses, train_acc)
            self.loss_tracker.log_epoch(epoch, "val", val_losses, val_acc)

            self.student_scheduler.step()

            # Print epoch summary
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}: Train {train_acc:.2f}%, Val {val_acc:.2f}%, Time {elapsed:.1f}s"
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

    def _train_disdkd(self, train_loader, val_loader):
        """
        Three-phase DisDKD training loop.

        Phase 1: Pretrain discriminator (teacher vs student noise)
        Phase 2: Adversarial feature alignment (student fools discriminator)
        Phase 3: DKD fine-tuning (pure logit distillation)
        """
        args = self.args
        best_acc = 0.0
        total_epochs = args.epochs

        # Phase boundaries
        phase1_max = getattr(args, "disdkd_phase1_epochs", 3)
        phase2_max = getattr(args, "disdkd_phase2_epochs", 7)
        phase1_min = getattr(args, "disdkd_phase1_min", 2)
        phase2_min = getattr(args, "disdkd_phase2_min", 3)
        disc_acc_threshold = getattr(args, "disdkd_disc_acc_threshold", 0.95)
        fool_rate_threshold = getattr(args, "disdkd_fool_rate_threshold", 0.85)

        # Learning rates per phase
        phase1_lr = getattr(args, "disdkd_phase1_lr", 1e-3)
        phase2_lr = getattr(args, "disdkd_phase2_lr", 1e-3)
        phase3_lr = getattr(args, "disdkd_phase3_lr", 1e-4)

        global_epoch = 0

        # ========== PHASE 1: Discriminator Warmup ==========
        print("\n" + "=" * 60)
        print("PHASE 1: Discriminator Warmup")
        print("=" * 60)

        self.model.set_phase(1)
        optimizer = self.model.get_phase1_optimizer(
            lr=phase1_lr, weight_decay=args.weight_decay
        )

        phase1_epochs_run = 0
        for epoch in range(phase1_max):
            start_time = time.time()

            losses, metrics = self._train_epoch_phase1(
                train_loader, optimizer, global_epoch
            )
            disc_acc = metrics["disc_accuracy"]

            # Log with phase info
            losses["disdkd_phase"] = 1
            losses["disc_acc"] = disc_acc * 100  # Convert to percentage
            self.loss_tracker.log_epoch(global_epoch, "train", losses, 0.0)

            elapsed = time.time() - start_time
            print(
                f'Epoch {global_epoch} [P1]: Disc Loss {losses["disc"]:.4f}, '
                f"Disc Acc {disc_acc:.2%}, Time {elapsed:.1f}s"
            )

            global_epoch += 1
            phase1_epochs_run += 1

            # Early transition check
            if disc_acc >= disc_acc_threshold and phase1_epochs_run >= phase1_min:
                print(
                    f"Discriminator converged (acc={disc_acc:.2%}), transitioning to Phase 2"
                )
                break

        print(f"Phase 1 completed after {phase1_epochs_run} epochs")

        # ========== PHASE 2: Adversarial Feature Alignment ==========
        print("\n" + "=" * 60)
        print("PHASE 2: Adversarial Feature Alignment")
        print("=" * 60)

        self.model.set_phase(2)
        optimizer = self.model.get_phase2_optimizer(
            lr=phase2_lr, weight_decay=args.weight_decay
        )

        phase2_epochs_run = 0
        remaining_for_phase2 = min(
            phase2_max, total_epochs - global_epoch - 5
        )  # Reserve at least 5 for Phase 3

        for epoch in range(remaining_for_phase2):
            start_time = time.time()

            losses, metrics = self._train_epoch_phase2(
                train_loader, optimizer, global_epoch
            )
            fool_rate = metrics["fool_rate"]

            # Log with phase info
            losses["disdkd_phase"] = 2
            losses["fool_rate"] = fool_rate * 100  # Convert to percentage
            self.loss_tracker.log_epoch(global_epoch, "train", losses, 0.0)

            elapsed = time.time() - start_time
            print(
                f'Epoch {global_epoch} [P2]: Adv Loss {losses["adversarial"]:.4f}, '
                f"Fool Rate {fool_rate:.2%}, Time {elapsed:.1f}s"
            )

            global_epoch += 1
            phase2_epochs_run += 1

            # Early transition check
            if fool_rate >= fool_rate_threshold and phase2_epochs_run >= phase2_min:
                print(
                    f"Feature alignment complete (fool_rate={fool_rate:.2%}), transitioning to Phase 3"
                )
                break

        print(f"Phase 2 completed after {phase2_epochs_run} epochs")

        # ========== PHASE 3: DKD Fine-tuning ==========
        print("\n" + "=" * 60)
        print("PHASE 3: DKD Fine-tuning")
        print("=" * 60)

        # Discard adversarial components
        self.model.discard_adversarial_components()
        self.model.set_phase(3)
        optimizer = self.model.get_phase3_optimizer(
            lr=phase3_lr, weight_decay=args.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

        remaining_epochs = total_epochs - global_epoch
        print(f"Phase 3 will run for {remaining_epochs} epochs")

        for epoch in range(remaining_epochs):
            start_time = time.time()

            losses, train_acc = self._train_epoch_phase3(
                train_loader, optimizer, global_epoch
            )
            val_losses, val_acc = self._validate(val_loader, global_epoch)

            # Log with phase info
            losses["disdkd_phase"] = 3
            val_losses["disdkd_phase"] = 3
            self.loss_tracker.log_epoch(global_epoch, "train", losses, train_acc)
            self.loss_tracker.log_epoch(global_epoch, "val", val_losses, val_acc)

            scheduler.step()

            elapsed = time.time() - start_time
            print(
                f"Epoch {global_epoch} [P3]: Train {train_acc:.2f}%, Val {val_acc:.2f}%, "
                f'DKD Loss {losses["dkd"]:.4f}, Time {elapsed:.1f}s'
            )

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    self.model,
                    optimizer,
                    global_epoch,
                    val_acc,
                    self.args,
                    is_best=True,
                )
                print(
                    f"New best accuracy: {best_acc:.2f}% - Saved to {self.args.save_dir}"
                )

            global_epoch += 1
            print("-" * 80)

        print(f"\nDisDKD Training Summary:")
        print(f"  Phase 1 (Discriminator): {phase1_epochs_run} epochs")
        print(f"  Phase 2 (Adversarial): {phase2_epochs_run} epochs")
        print(f"  Phase 3 (DKD): {remaining_epochs} epochs")
        print(f"  Total: {global_epoch} epochs")

        return best_acc

    def _train_epoch_phase1(self, train_loader, optimizer, epoch):
        """Train discriminator for one epoch (Phase 1)."""
        self.model.train()

        disc_loss_meter = AverageMeter()
        disc_acc_meter = AverageMeter()
        teacher_pred_meter = AverageMeter()
        student_pred_meter = AverageMeter()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Phase 1]", leave=False)

        for inputs, _ in progress_bar:
            inputs = inputs.to(self.device)
            batch_size = inputs.size(0)

            optimizer.zero_grad()
            result = self.model(inputs)

            loss = result["disc_loss"]
            loss.backward()
            optimizer.step()

            # Update meters
            disc_loss_meter.update(loss.item(), batch_size)
            disc_acc_meter.update(result["disc_accuracy"], batch_size)
            teacher_pred_meter.update(result["teacher_pred_mean"], batch_size)
            student_pred_meter.update(result["student_pred_mean"], batch_size)

            progress_bar.set_postfix(
                {
                    "loss": f"{disc_loss_meter.avg:.4f}",
                    "acc": f"{disc_acc_meter.avg:.2%}",
                    "T": f"{teacher_pred_meter.avg:.2f}",
                    "S": f"{student_pred_meter.avg:.2f}",
                }
            )

        progress_bar.close()

        losses = {"disc": disc_loss_meter.avg, "total": disc_loss_meter.avg}
        metrics = {"disc_accuracy": disc_acc_meter.avg}

        return losses, metrics

    def _train_epoch_phase2(self, train_loader, optimizer, epoch):
        """Train student (up to layer G) for one epoch (Phase 2)."""
        self.model.train()

        adv_loss_meter = AverageMeter()
        fool_rate_meter = AverageMeter()
        student_pred_meter = AverageMeter()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Phase 2]", leave=False)

        for inputs, _ in progress_bar:
            inputs = inputs.to(self.device)
            batch_size = inputs.size(0)

            optimizer.zero_grad()
            result = self.model(inputs)

            loss = result["adversarial_loss"]
            loss.backward()
            optimizer.step()

            # Update meters
            adv_loss_meter.update(loss.item(), batch_size)
            fool_rate_meter.update(result["fool_rate"], batch_size)
            student_pred_meter.update(result["student_pred_mean"], batch_size)

            progress_bar.set_postfix(
                {
                    "adv_loss": f"{adv_loss_meter.avg:.4f}",
                    "fool": f"{fool_rate_meter.avg:.2%}",
                    "S_pred": f"{student_pred_meter.avg:.2f}",
                }
            )

        progress_bar.close()

        losses = {"adversarial": adv_loss_meter.avg, "total": adv_loss_meter.avg}
        metrics = {"fool_rate": fool_rate_meter.avg}

        return losses, metrics

    def _train_epoch_phase3(self, train_loader, optimizer, epoch):
        """Train full student with DKD for one epoch (Phase 3)."""
        self.model.train()

        ce_loss_meter = AverageMeter()
        dkd_loss_meter = AverageMeter()
        total_loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Phase 3]", leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)

            optimizer.zero_grad()
            result = self.model(inputs, targets)

            student_logits = result["student_logits"]
            dkd_loss = result["dkd_loss"]

            # CE + DKD
            ce_loss = self.criterion(student_logits, targets)
            total_loss = self.args.alpha * ce_loss + dkd_loss

            total_loss.backward()
            optimizer.step()

            # Compute accuracy
            acc = accuracy(student_logits, targets, topk=(1,))[0]

            # Update meters
            ce_loss_meter.update(ce_loss.item(), batch_size)
            dkd_loss_meter.update(dkd_loss.item(), batch_size)
            total_loss_meter.update(total_loss.item(), batch_size)
            accuracy_meter.update(acc.item(), batch_size)

            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss_meter.avg:.4f}",
                    "ce": f"{ce_loss_meter.avg:.4f}",
                    "dkd": f"{dkd_loss_meter.avg:.4f}",
                    "acc": f"{accuracy_meter.avg:.2f}%",
                }
            )

        progress_bar.close()

        losses = {
            "total": total_loss_meter.avg,
            "ce": ce_loss_meter.avg,
            "dkd": dkd_loss_meter.avg,
        }

        return losses, accuracy_meter.avg

    def _train_epoch_standard(self, train_loader, epoch):
        """Train for one epoch (standard methods)."""
        self.model.train()
        meters = self._init_meters()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            # Handle different data formats
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
        return self._get_average_losses(meters)

    def _validate(self, val_loader, epoch=None):
        """Validate the model."""
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()

        desc = f"Epoch {epoch} Val" if epoch is not None else "Validation"
        progress_bar = tqdm(val_loader, desc=desc, leave=False)

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                if self.args.method == "DisDKD":
                    if self.model.current_phase == 3:
                        result = self.model(inputs, targets)
                        student_logits = result["student_logits"]
                    else:
                        # During Phase 1 & 2, just do a forward pass for logits
                        student_logits = self.model.student(inputs)
                elif self.args.method in ["DKD"]:
                    _, student_logits, _ = self.model(inputs, targets)
                else:
                    _, student_logits, _ = self.model(inputs)

                loss = self.criterion(student_logits, targets)
                acc1 = accuracy(student_logits, targets, topk=(1,))[0]

                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1.item(), inputs.size(0))

                progress_bar.set_postfix(
                    {"val_loss": f"{losses.avg:.4f}", "val_acc": f"{top1.avg:.2f}%"}
                )

        progress_bar.close()
        return {"total": losses.avg}, top1.avg

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

        # For DKD, the method_specific_loss already contains the full distillation loss
        if self.args.method in ["DKD"]:
            kd_loss = torch.tensor(0.0, device=student_logits.device)
        else:
            kd_loss = self._compute_kd_loss(teacher_logits, student_logits)

        # Weighted losses
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

        # Method-specific meters
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

    def forward(self, x):
        return self.model(x)


class Trainer:
    """
    Unified trainer for all knowledge distillation methods.
    Handles interleaved adversarial training for DisDKD.
    """

    def __init__(self, teacher, student, num_classes, criterion, loss_tracker, device, args):
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.criterion = criterion
        self.loss_tracker = loss_tracker
        self.device = device
        self.args = args
        self.method = args.method

        # Build distillation model
        if self.method != "Pretraining":
            self.distill_model = self._build_distillation_model()
        else:
            self.distill_model = None

        # Setup optimizers
        self._setup_optimizers()

        # Track best accuracy
        self.best_val_acc = 0.0

    def _build_distillation_model(self):
        """Build the distillation model based on method."""
        from utils.model_factory import create_distillation_model

        return create_distillation_model(
            self.args,
            self.teacher,
            self.student,
            self.num_classes
        ).to(self.device)

    def _setup_optimizers(self):
        """Setup optimizers based on method."""
        if self.method == "Pretraining":
            # Only train teacher
            self.optimizer = self._get_optimizer(self.teacher.parameters())
            self.scheduler = self._get_scheduler(self.optimizer)
        elif self.method == "DisDKD":
            # For DisDKD, we'll create optimizers per phase
            # Phase 1: discriminator and generator optimizers
            self.optimizer_D = self.distill_model.get_discriminator_optimizer(
                lr=self.args.disdkd_phase1_lr,
                weight_decay=self.args.weight_decay
            )
            self.optimizer_G = self.distill_model.get_generator_optimizer(
                lr=self.args.disdkd_phase2_lr,  # generator LR
                weight_decay=self.args.weight_decay
            )
            # Phase 2: DKD optimizer (created later when transitioning to Phase 2)
            self.optimizer_DKD = None
            self.scheduler = None  # We'll manage LR manually for DisDKD
        else:
            # Standard methods
            self.optimizer = self._get_optimizer(self.student.parameters())
            self.scheduler = self._get_scheduler(self.optimizer)

    def _get_optimizer(self, params):
        """Get optimizer based on args."""
        if self.args.optimizer == "adam":
            return torch.optim.Adam(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

    def _get_scheduler(self, optimizer):
        """Get learning rate scheduler."""
        if self.args.step_size > 0:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.args.step_size,
                gamma=self.args.lr_decay
            )
        return None

    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training with method: {self.method}")
        print(f"{'='*60}\n")

        if self.method == "DisDKD":
            return self._train_disdkd(train_loader, val_loader)
        else:
            return self._train_standard(train_loader, val_loader)

    def _train_disdkd(self, train_loader, val_loader):
        """
        Training loop for DisDKD with interleaved adversarial training.

        Phase 1 (first N epochs): Interleaved D/G training
        Phase 2 (remaining epochs): DKD fine-tuning
        """
        phase1_epochs = self.args.disdkd_phase1_epochs
        total_epochs = self.args.epochs
        k_disc_steps = self.args.disdkd_k_disc_steps if hasattr(self.args, 'disdkd_k_disc_steps') else 1

        # Phase 1: Adversarial (interleaved D/G)
        print(f"\n{'='*60}")
        print(f"PHASE 1: Adversarial Training (Interleaved D/G)")
        print(f"  Epochs: {phase1_epochs}")
        print(f"  Discriminator steps per batch: {k_disc_steps}")
        print(f"  Generator steps per batch: 1")
        print(f"  Discriminator LR: {self.args.disdkd_phase1_lr}")
        print(f"  Generator LR: {self.args.disdkd_phase2_lr}")
        print(f"{'='*60}\n")

        self.distill_model.set_phase(1)

        for epoch in range(1, phase1_epochs + 1):
            train_metrics = self._train_epoch_phase1(train_loader, epoch, k_disc_steps)
            val_metrics = self._validate_epoch_phase1(val_loader, epoch)

            # Log metrics
            self.loss_tracker.log_epoch(
                epoch=epoch,
                phase="train",
                losses={
                    "disdkd_phase": 1,
                    "disc_loss": train_metrics["disc_loss"],
                    "disc_acc": train_metrics["disc_acc"],
                    "gen_loss": train_metrics["gen_loss"],
                    "fool_rate": train_metrics["fool_rate"],
                    "teacher_pred_mean": train_metrics["teacher_pred_mean"],
                    "student_pred_mean": train_metrics["student_pred_mean"],
                },
                accuracy=0.0  # No classification accuracy in Phase 1
            )

            self.loss_tracker.log_epoch(
                epoch=epoch,
                phase="val",
                losses={
                    "disdkd_phase": 1,
                    "disc_loss": val_metrics["disc_loss"],
                    "disc_acc": val_metrics["disc_acc"],
                    "gen_loss": val_metrics["gen_loss"],
                    "fool_rate": val_metrics["fool_rate"],
                    "teacher_pred_mean": val_metrics["teacher_pred_mean"],
                    "student_pred_mean": val_metrics["student_pred_mean"],
                },
                accuracy=0.0
            )

            print(f"Epoch {epoch}/{phase1_epochs} [Phase 1] - "
                  f"D_loss: {train_metrics['disc_loss']:.4f}, "
                  f"D_acc: {train_metrics['disc_acc']:.2f}%, "
                  f"G_loss: {train_metrics['gen_loss']:.4f}, "
                  f"Fool_rate: {train_metrics['fool_rate']:.2f}%")

        # Transition to Phase 2
        print(f"\n{'='*60}")
        print(f"PHASE 2: DKD Fine-tuning")
        print(f"  Epochs: {total_epochs - phase1_epochs}")
        print(f"  Learning rate: {self.args.disdkd_phase3_lr}")
        print(f"{'='*60}\n")

        self.distill_model.set_phase(2)
        self.distill_model.discard_adversarial_components()

        # Create DKD optimizer
        self.optimizer_DKD = self.distill_model.get_dkd_optimizer(
            lr=self.args.disdkd_phase3_lr,
            weight_decay=self.args.weight_decay
        )

        # Phase 2: DKD
        for epoch in range(phase1_epochs + 1, total_epochs + 1):
            train_metrics = self._train_epoch_phase2(train_loader, epoch)
            val_metrics = self._validate_epoch_phase2(val_loader, epoch)

            # Log metrics
            self.loss_tracker.log_epoch(
                epoch=epoch,
                phase="train",
                losses={
                    "disdkd_phase": 2,
                    "ce_loss": train_metrics["ce_loss"],
                    "dkd_loss": train_metrics["dkd_loss"],
                    "total_loss": train_metrics["total_loss"],
                },
                accuracy=train_metrics["accuracy"]
            )

            self.loss_tracker.log_epoch(
                epoch=epoch,
                phase="val",
                losses={
                    "disdkd_phase": 2,
                    "ce_loss": val_metrics["ce_loss"],
                    "dkd_loss": val_metrics["dkd_loss"],
                    "total_loss": val_metrics["total_loss"],
                },
                accuracy=val_metrics["accuracy"]
            )

            # Track best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self._save_checkpoint(epoch, is_best=True)

            print(f"Epoch {epoch}/{total_epochs} [Phase 2] - "
                  f"Loss: {train_metrics['total_loss']:.4f}, "
                  f"Train_acc: {train_metrics['accuracy']:.2f}%, "
                  f"Val_acc: {val_metrics['accuracy']:.2f}%")

        return self.best_val_acc

    def _train_epoch_phase1(self, train_loader, epoch, k_disc_steps):
        """Train one epoch of Phase 1 (interleaved D/G)."""
        self.distill_model.train()

        metrics = {
            "disc_loss": 0.0,
            "disc_acc": 0.0,
            "gen_loss": 0.0,
            "fool_rate": 0.0,
            "teacher_pred_mean": 0.0,
            "student_pred_mean": 0.0,
        }
        num_batches = 0

        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(self.device)

            # Train discriminator for k steps
            for _ in range(k_disc_steps):
                self.distill_model.set_discriminator_mode()
                self.optimizer_D.zero_grad()

                outputs = self.distill_model(inputs, mode='discriminator')
                loss_D = outputs['disc_loss']
                loss_D.backward()
                self.optimizer_D.step()

                metrics["disc_loss"] += outputs["disc_loss"].item()
                metrics["disc_acc"] += outputs["disc_accuracy"] * 100
                metrics["teacher_pred_mean"] += outputs["teacher_pred_mean"]

            # Train generator for 1 step
            self.distill_model.set_generator_mode()
            self.optimizer_G.zero_grad()

            outputs = self.distill_model(inputs, mode='generator')
            loss_G = outputs['gen_loss']
            loss_G.backward()
            self.optimizer_G.step()

            metrics["gen_loss"] += outputs["gen_loss"].item()
            metrics["fool_rate"] += outputs["fool_rate"] * 100
            metrics["student_pred_mean"] += outputs["student_pred_mean"]

            num_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
            if key == "disc_loss" or key == "disc_acc" or key == "teacher_pred_mean":
                metrics[key] /= k_disc_steps  # Account for multiple D steps

        return metrics

    def _validate_epoch_phase1(self, val_loader, epoch):
        """Validate one epoch of Phase 1."""
        self.distill_model.eval()

        metrics = {
            "disc_loss": 0.0,
            "disc_acc": 0.0,
            "gen_loss": 0.0,
            "fool_rate": 0.0,
            "teacher_pred_mean": 0.0,
            "student_pred_mean": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(self.device)

                # Evaluate discriminator
                self.distill_model.set_discriminator_mode()
                disc_outputs = self.distill_model(inputs, mode='discriminator')

                # Evaluate generator
                self.distill_model.set_generator_mode()
                gen_outputs = self.distill_model(inputs, mode='generator')

                metrics["disc_loss"] += disc_outputs["disc_loss"].item()
                metrics["disc_acc"] += disc_outputs["disc_accuracy"] * 100
                metrics["teacher_pred_mean"] += disc_outputs["teacher_pred_mean"]
                metrics["gen_loss"] += gen_outputs["gen_loss"].item()
                metrics["fool_rate"] += gen_outputs["fool_rate"] * 100
                metrics["student_pred_mean"] += gen_outputs["student_pred_mean"]

                num_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        return metrics

    def _train_epoch_phase2(self, train_loader, epoch):
        """Train one epoch of Phase 2 (DKD)."""
        self.distill_model.train()

        metrics = {
            "ce_loss": 0.0,
            "dkd_loss": 0.0,
            "total_loss": 0.0,
            "accuracy": 0.0,
        }
        num_batches = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer_DKD.zero_grad()

            outputs = self.distill_model(inputs, targets=targets)
            student_logits = outputs["student_logits"]
            dkd_loss = outputs["dkd_loss"]
            ce_loss = self.criterion(student_logits, targets)

            # Total loss: CE + DKD
            total_loss = self.args.alpha * ce_loss + self.args.beta * dkd_loss
            total_loss.backward()
            self.optimizer_DKD.step()

            # Compute accuracy
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            metrics["ce_loss"] += ce_loss.item()
            metrics["dkd_loss"] += dkd_loss.item()
            metrics["total_loss"] += total_loss.item()
            num_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        metrics["accuracy"] = 100.0 * correct / total

        return metrics

    def _validate_epoch_phase2(self, val_loader, epoch):
        """Validate one epoch of Phase 2."""
        self.distill_model.eval()

        metrics = {
            "ce_loss": 0.0,
            "dkd_loss": 0.0,
            "total_loss": 0.0,
            "accuracy": 0.0,
        }
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.distill_model(inputs, targets=targets)
                student_logits = outputs["student_logits"]
                dkd_loss = outputs["dkd_loss"]
                ce_loss = self.criterion(student_logits, targets)

                total_loss = self.args.alpha * ce_loss + self.args.beta * dkd_loss

                # Compute accuracy
                _, predicted = student_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                metrics["ce_loss"] += ce_loss.item()
                metrics["dkd_loss"] += dkd_loss.item()
                metrics["total_loss"] += total_loss.item()
                num_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        metrics["accuracy"] = 100.0 * correct / total

        return metrics

    def _train_standard(self, train_loader, val_loader):
        """Standard training loop for other methods."""
        # TODO: Implement standard training for other methods
        raise NotImplementedError("Standard training not yet implemented. Use DisDKD for now.")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        from utils.checkpoint import save_checkpoint

        save_checkpoint(
            self.student.state_dict() if self.student else self.teacher.state_dict(),
            self.args.save_dir,
            self.args.dataset,
            self.args.train_domains if hasattr(self.args, 'train_domains') else None,
            self.args.val_domains if hasattr(self.args, 'val_domains') else None,
            is_best=is_best
        )
