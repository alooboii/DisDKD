import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from distill.direct_trajectory_flow import (
    DirectTrajectoryVelocity,
    compute_end_loss,
    compute_fm_loss,
    compute_path_loss,
    generate_teacher_trajectory_targets,
)
from utils.utils import accuracy


class CSVLogger:
    def __init__(self, path: Path, headers: Sequence[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = list(headers)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()

    def write(self, row: Dict):
        out = {k: row.get(k, "") for k in self.headers}
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(out)


class OrderedHooks:
    def __init__(self, named_modules: Iterable[Tuple[str, nn.Module]]):
        self.names = []
        self.features: Dict[str, torch.Tensor] = {}
        self._handles = []

        def make_hook(name):
            def _hook(_module, _input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.features[name] = output

            return _hook

        for name, module in named_modules:
            self.names.append(name)
            self._handles.append(module.register_forward_hook(make_hook(name)))

    def clear(self):
        self.features.clear()

    def ordered(self) -> List[torch.Tensor]:
        return [self.features[name] for name in self.names]

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class DirectTrajectoryRunner:
    """DirectTrajectoryZFlow runner for ViT flow + student trajectory distillation."""

    def __init__(self, args, teacher, student, train_loader, val_loader, device):
        self.args = args
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self._freeze_module(self.teacher)
        self.teacher.eval()

        self.teacher_block_names, self.teacher_block_modules = self._extract_vit_blocks(
            self.teacher.model, role="teacher"
        )
        self.teacher_hooks = OrderedHooks(
            zip(self.teacher_block_names, self.teacher_block_modules)
        )

        self.student_hooks: Optional[OrderedHooks] = None
        self.student_block_names: List[str] = []
        self.student_block_modules: List[nn.Module] = []

        self.start_layer_1b, self.end_layer_1b, self.intermediate_layers_1b = self._resolve_trajectory_layers(
            total_layers=len(self.teacher_block_modules)
        )

        self.flow: Optional[DirectTrajectoryVelocity] = None
        self.student_adapters = nn.ModuleDict()

        self.save_root = Path(args.save_dir)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.traj_dir = self.save_root / "direct_trajectory_zflow"
        self.traj_dir.mkdir(parents=True, exist_ok=True)

        self._printed_teacher_shapes = False
        self._printed_student_shapes = False

    def close(self):
        if self.teacher_hooks is not None:
            self.teacher_hooks.remove()
        if self.student_hooks is not None:
            self.student_hooks.remove()

    def run(self):
        try:
            if self.args.traj_stage == "flow":
                self.train_flow_stage()
                return

            if self.args.traj_stage == "student":
                self.train_student_stage()
                return

            if self.args.traj_stage == "all":
                flow_ckpt = self.train_flow_stage()
                self.args.trajectory_ckpt = str(flow_ckpt)
                self.train_student_stage()
                return

            raise ValueError(
                f"Unknown traj_stage '{self.args.traj_stage}'. Use flow|student|all"
            )
        finally:
            self.close()

    def _freeze_module(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def _extract_inputs_targets(self, batch):
        if len(batch) == 3:
            inputs, targets, _ = batch
        else:
            inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)

    def _extract_vit_blocks(self, model: nn.Module, role: str) -> Tuple[List[str], List[nn.Module]]:
        if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
            raise ValueError(
                f"DirectTrajectoryZFlow currently supports torchvision ViT models only. "
                f"{role} model is missing encoder.layers"
            )

        layers = model.encoder.layers
        named = list(layers.named_children())
        if len(named) == 0:
            raise ValueError(f"{role} ViT encoder.layers has no child blocks")

        names = [name for name, _ in named]
        modules = [module for _, module in named]
        return names, modules

    def _resolve_trajectory_layers(self, total_layers: int) -> Tuple[int, int, List[int]]:
        start = int(self.args.traj_start_layer)
        end = total_layers if int(self.args.traj_end_layer) == -1 else int(self.args.traj_end_layer)

        if start < 1 or start > total_layers:
            raise ValueError(
                f"Invalid --traj_start_layer={start}; expected 1..{total_layers}"
            )
        if end < 1 or end > total_layers:
            raise ValueError(f"Invalid --traj_end_layer={end}; expected 1..{total_layers}")
        if start >= end:
            raise ValueError(
                f"Trajectory start must be < end. Got start={start}, end={end}"
            )

        if self.args.traj_intermediate_layers:
            mids = sorted(set(int(v) for v in self.args.traj_intermediate_layers))
        else:
            mids = list(range(start + 1, end))

        filtered = []
        for m in mids:
            if m <= start or m >= end:
                continue
            if m < 1 or m > total_layers:
                raise ValueError(
                    f"Invalid intermediate layer {m}; expected within 1..{total_layers}"
                )
            filtered.append(m)

        return start, end, filtered

    def _teacher_forward(self, inputs: torch.Tensor):
        self.teacher_hooks.clear()
        logits = self.teacher(inputs)
        hidden = self.teacher_hooks.ordered()
        if len(hidden) != len(self.teacher_block_names):
            raise RuntimeError(
                f"Teacher hook capture mismatch: captured {len(hidden)} blocks, expected {len(self.teacher_block_names)}"
            )
        hidden = [h.detach() for h in hidden]

        if not self._printed_teacher_shapes:
            print("[DirectTrajectoryZFlow] Teacher block shapes:")
            for i, (name, h) in enumerate(zip(self.teacher_block_names, hidden), start=1):
                print(f"  L{i:02d} {name}: {tuple(h.shape)}")
            self._printed_teacher_shapes = True

        return logits.detach(), hidden

    def _student_forward(self, inputs: torch.Tensor):
        if self.student is None or self.student_hooks is None:
            raise RuntimeError("Student hooks are not initialized")

        self.student_hooks.clear()
        logits = self.student(inputs)
        hidden = self.student_hooks.ordered()
        if len(hidden) != len(self.student_block_names):
            raise RuntimeError(
                f"Student hook capture mismatch: captured {len(hidden)} blocks, expected {len(self.student_block_names)}"
            )

        if not self._printed_student_shapes:
            print("[DirectTrajectoryZFlow] Student block shapes:")
            for i, (name, h) in enumerate(zip(self.student_block_names, hidden), start=1):
                print(f"  S{i:02d} {name}: {tuple(h.shape)}")
            self._printed_student_shapes = True

        return logits, hidden

    def _ensure_flow(self, hidden_dim: int):
        if self.flow is not None:
            if self.flow.hidden_dim != hidden_dim:
                raise ValueError(
                    f"Flow hidden_dim mismatch: existing={self.flow.hidden_dim}, requested={hidden_dim}"
                )
            return

        self.flow = DirectTrajectoryVelocity(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=self.args.traj_velocity_hidden_dim,
            num_blocks=self.args.traj_velocity_num_blocks,
            use_attention=self.args.traj_velocity_use_attention,
            num_heads=self.args.traj_velocity_num_heads,
            time_embed_dim=self.args.traj_time_embed_dim,
        ).to(self.device)

    def _path_targets_from_teacher(self, teacher_hidden: List[torch.Tensor]):
        start = self.start_layer_1b
        end = self.end_layer_1b
        denom = float(end - start)

        targets = []
        for layer_idx in self.intermediate_layers_1b:
            tau = (layer_idx - start) / denom
            targets.append((tau, teacher_hidden[layer_idx - 1], layer_idx))
        return targets

    def _assert_teacher_traj_dim_consistency(self, teacher_hidden: List[torch.Tensor]):
        indices = [self.start_layer_1b, self.end_layer_1b] + self.intermediate_layers_1b
        dims = [teacher_hidden[i - 1].shape[-1] for i in indices]
        if len(set(dims)) != 1:
            raise ValueError(
                f"Teacher trajectory layers must share hidden dim for v1. Found dims={dims} at layers={indices}"
            )

    def _kl_to_teacher(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        t = self.args.tau
        return F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.softmax(teacher_logits / t, dim=1),
            reduction="batchmean",
        ) * (t * t)

    def _save_flow_checkpoint(self, path: Path, epoch: int, metric: float, optimizer):
        payload = {
            "epoch": epoch,
            "metric": metric,
            "flow_state_dict": self.flow.state_dict() if self.flow is not None else None,
            "teacher_block_names": self.teacher_block_names,
            "traj_start_layer": self.start_layer_1b,
            "traj_end_layer": self.end_layer_1b,
            "traj_intermediate_layers": self.intermediate_layers_1b,
            "flow_hidden_dim": self.flow.hidden_dim if self.flow is not None else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "args": vars(self.args),
        }
        torch.save(payload, path)

    def _load_flow_checkpoint(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        hidden_dim = ckpt.get("flow_hidden_dim", None)
        if hidden_dim is None:
            raise ValueError(
                f"Checkpoint {ckpt_path} is missing 'flow_hidden_dim'."
            )
        self._ensure_flow(int(hidden_dim))
        self.flow.load_state_dict(ckpt["flow_state_dict"], strict=False)

    def train_flow_stage(self) -> Path:
        print("\n=== DirectTrajectoryZFlow Stage: Teacher Trajectory Flow ===")

        warm_batch = next(iter(self.train_loader))
        warm_inputs, _ = self._extract_inputs_targets(warm_batch)
        with torch.no_grad():
            _, warm_hidden = self._teacher_forward(warm_inputs)

        self._assert_teacher_traj_dim_consistency(warm_hidden)
        hidden_dim = warm_hidden[self.start_layer_1b - 1].shape[-1]
        self._ensure_flow(hidden_dim)
        self.flow.train()

        optimizer = torch.optim.Adam(
            self.flow.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.args.step_size, gamma=self.args.lr_decay
        )

        path_layer_headers = [f"path_layer_{l}" for l in self.intermediate_layers_1b]
        logger = CSVLogger(
            self.traj_dir / "trajectory_flow_log.csv",
            headers=[
                "epoch",
                "phase",
                "total",
                "fm",
                "path",
                "end",
                "endpoint_error",
                "teacher_top1",
                "lr",
            ]
            + path_layer_headers,
        )

        best_val = float("inf")
        best_ckpt = self.traj_dir / "trajectory_flow_best.pth"
        last_ckpt = self.traj_dir / "trajectory_flow_last.pth"

        for epoch in range(self.args.epochs):
            start_t = time.time()
            train_metrics = self._run_flow_epoch(train=True, optimizer=optimizer)
            val_metrics = self._run_flow_epoch(train=False, optimizer=None)

            lr = optimizer.param_groups[0]["lr"]
            logger.write({"epoch": epoch, "phase": "train", "lr": lr, **train_metrics})
            logger.write({"epoch": epoch, "phase": "val", "lr": lr, **val_metrics})
            scheduler.step()

            elapsed = time.time() - start_t
            print(
                f"Epoch {epoch}: flow train total={train_metrics['total']:.4f}, "
                f"val total={val_metrics['total']:.4f}, "
                f"val fm/path/end={val_metrics['fm']:.4f}/{val_metrics['path']:.4f}/{val_metrics['end']:.4f}, "
                f"time={elapsed:.1f}s"
            )

            self._save_flow_checkpoint(last_ckpt, epoch, val_metrics["total"], optimizer)
            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                self._save_flow_checkpoint(best_ckpt, epoch, best_val, optimizer)
                print(f"[DirectTrajectoryZFlow] New best flow checkpoint: {best_ckpt}")

        print(f"[DirectTrajectoryZFlow] Flow stage complete. Best ckpt: {best_ckpt}")
        return best_ckpt

    def _run_flow_epoch(self, train: bool, optimizer):
        if self.flow is None:
            raise RuntimeError("Flow model is not initialized")

        self.flow.train(mode=train)
        meter_sums = defaultdict(float)
        sample_count = 0

        loader = self.train_loader if train else self.val_loader
        pbar = tqdm(loader, desc="Traj Flow Train" if train else "Traj Flow Val", leave=False)

        for batch_idx, batch in enumerate(pbar):
            inputs, targets = self._extract_inputs_targets(batch)
            bsz = inputs.size(0)

            with torch.no_grad():
                teacher_logits, teacher_hidden = self._teacher_forward(inputs)

            h_start = teacher_hidden[self.start_layer_1b - 1]
            h_end = teacher_hidden[self.end_layer_1b - 1]
            path_targets = self._path_targets_from_teacher(teacher_hidden)

            if train:
                optimizer.zero_grad()

            grad_ctx = torch.enable_grad() if train else torch.no_grad()
            with grad_ctx:
                fm_loss = compute_fm_loss(self.flow, h_start, h_end)
                path_loss, path_detail = compute_path_loss(
                    self.flow,
                    h_start,
                    [(tau, h) for tau, h, _ in path_targets],
                    steps=self.args.traj_flow_steps,
                )
                end_loss, h_hat_end = compute_end_loss(
                    self.flow,
                    h_start,
                    h_end,
                    steps=self.args.traj_flow_steps,
                )

                total_loss = (
                    self.args.traj_lambda_fm * fm_loss
                    + self.args.traj_lambda_path * path_loss
                    + self.args.traj_lambda_end * end_loss
                )

            if train:
                total_loss.backward()
                optimizer.step()

            teacher_top1 = accuracy(teacher_logits, targets, topk=(1,))[0].item()
            endpoint_error = F.mse_loss(h_hat_end, h_end).item()

            metrics = {
                "total": total_loss.item(),
                "fm": fm_loss.item(),
                "path": path_loss.item(),
                "end": end_loss.item(),
                "endpoint_error": endpoint_error,
                "teacher_top1": teacher_top1,
            }

            for tau, _, layer_idx in path_targets:
                key = f"path_tau_{tau:.4f}"
                metrics[f"path_layer_{layer_idx}"] = path_detail.get(key, 0.0)

            for key, value in metrics.items():
                meter_sums[key] += float(value) * bsz
            sample_count += bsz

            if batch_idx % max(1, self.args.print_freq // 10) == 0:
                pbar.set_postfix(
                    {
                        "total": f"{meter_sums['total']/max(sample_count,1):.4f}",
                        "fm": f"{meter_sums['fm']/max(sample_count,1):.4f}",
                        "path": f"{meter_sums['path']/max(sample_count,1):.4f}",
                    }
                )

        return {k: v / max(sample_count, 1) for k, v in meter_sums.items()}

    def _init_student_hooks(self):
        if self.student is None:
            raise ValueError(
                "DirectTrajectoryZFlow student stage requires --student and a valid student model"
            )

        self.student_block_names, self.student_block_modules = self._extract_vit_blocks(
            self.student.model, role="student"
        )
        self.student_hooks = OrderedHooks(
            zip(self.student_block_names, self.student_block_modules)
        )

    def _align_student_hidden(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        adapter: Optional[nn.Module],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = student_h
        t = teacher_h

        if self.args.traj_match_tokens == "cls":
            s = s[:, 0, :]
            t = t[:, 0, :]
            if adapter is not None:
                s = adapter(s)
            return s, t

        if s.size(1) != t.size(1):
            s = s.transpose(1, 2)
            s = F.interpolate(s, size=t.size(1), mode="linear", align_corners=False)
            s = s.transpose(1, 2)

        if adapter is not None:
            s = adapter(s)

        return s, t

    def _build_adapter(self, in_dim: int, out_dim: int) -> nn.Module:
        if self.args.traj_student_adapter_type == "linear":
            return nn.Linear(in_dim, out_dim)

        hidden = max(in_dim, out_dim)
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def _prepare_adapters(self, student_hidden: List[torch.Tensor], teacher_targets: List[torch.Tensor]):
        for k, (s, t) in enumerate(zip(student_hidden, teacher_targets), start=1):
            key = f"layer_{k}"
            in_dim = s.shape[-1]
            out_dim = t.shape[-1]
            if in_dim == out_dim:
                continue

            if not self.args.traj_student_adapter:
                raise ValueError(
                    f"Student/teacher hidden dim mismatch at layer {k}: {in_dim} vs {out_dim}. "
                    "Enable --traj_student_adapter true."
                )

            if key not in self.student_adapters:
                self.student_adapters[key] = self._build_adapter(in_dim, out_dim).to(self.device)

    def _build_discrete_targets(
        self, teacher_hidden: List[torch.Tensor], student_num_layers: int
    ) -> List[torch.Tensor]:
        start = self.start_layer_1b
        end = self.end_layer_1b

        targets = []
        for k in range(1, student_num_layers + 1):
            depth = k / float(student_num_layers)
            mapped = int(round(start + depth * (end - start)))
            mapped = max(start, min(mapped, end))
            targets.append(teacher_hidden[mapped - 1])
        return targets

    def _load_or_require_flow_for_student(self):
        if self.args.traj_target_mode == "discrete":
            return

        if not self.args.trajectory_ckpt:
            raise ValueError(
                "--trajectory_ckpt is required for student stage when --traj_target_mode flow"
            )

        self._load_flow_checkpoint(Path(self.args.trajectory_ckpt))
        if self.flow is None:
            raise RuntimeError("Failed to initialize flow from checkpoint")

        self._freeze_module(self.flow)
        self.flow.eval()

    def train_student_stage(self):
        print("\n=== DirectTrajectoryZFlow Stage: Student Distillation ===")

        self._init_student_hooks()
        self._load_or_require_flow_for_student()

        if self.args.traj_target_mode == "flow":
            if self.flow is None:
                raise RuntimeError("Flow must be loaded for flow target mode")

        warm_batch = next(iter(self.train_loader))
        warm_inputs, _ = self._extract_inputs_targets(warm_batch)
        with torch.no_grad():
            _, teacher_hidden = self._teacher_forward(warm_inputs)
            _, student_hidden = self._student_forward(warm_inputs)

            if self.args.traj_target_mode == "flow":
                h_start = teacher_hidden[self.start_layer_1b - 1]
                teacher_targets = generate_teacher_trajectory_targets(
                    h_start=h_start,
                    velocity_model=self.flow,
                    student_num_layers=len(student_hidden),
                    flow_steps=self.args.traj_flow_steps,
                    start_t=0.0,
                    end_t=1.0,
                )
            else:
                teacher_targets = self._build_discrete_targets(
                    teacher_hidden, len(student_hidden)
                )
            if len(teacher_targets) != len(student_hidden):
                raise RuntimeError(
                    f"Target count mismatch: got {len(teacher_targets)} targets for {len(student_hidden)} student layers"
                )

            self._prepare_adapters(student_hidden, teacher_targets)

        params = list(self.student.parameters()) + list(self.student_adapters.parameters())
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.args.step_size, gamma=self.args.lr_decay
        )

        student_layer_headers = [f"traj_layer_{k}" for k in range(1, len(student_hidden) + 1)]
        logger = CSVLogger(
            self.traj_dir / "trajectory_student_log.csv",
            headers=[
                "epoch",
                "phase",
                "total",
                "ce",
                "kd",
                "traj",
                "student_top1",
                "teacher_top1",
                "kl_to_teacher",
                "lr",
            ]
            + student_layer_headers,
        )

        eval_logger = CSVLogger(
            self.traj_dir / "trajectory_eval_log.csv",
            headers=[
                "epoch",
                "target_mode",
                "student_accuracy",
                "teacher_accuracy",
                "kl_to_teacher",
                "hidden_alignment_error",
            ],
        )

        best_val = -1.0
        best_ckpt = self.traj_dir / "trajectory_student_best.pth"
        last_ckpt = self.traj_dir / "trajectory_student_last.pth"

        for epoch in range(self.args.epochs):
            start_t = time.time()
            train_metrics = self._run_student_epoch(train=True, optimizer=optimizer)
            val_metrics = self._run_student_epoch(train=False, optimizer=None)

            lr = optimizer.param_groups[0]["lr"]
            logger.write({"epoch": epoch, "phase": "train", "lr": lr, **train_metrics})
            logger.write({"epoch": epoch, "phase": "val", "lr": lr, **val_metrics})
            eval_logger.write(
                {
                    "epoch": epoch,
                    "target_mode": self.args.traj_target_mode,
                    "student_accuracy": val_metrics.get("student_top1", 0.0),
                    "teacher_accuracy": val_metrics.get("teacher_top1", 0.0),
                    "kl_to_teacher": val_metrics.get("kl_to_teacher", 0.0),
                    "hidden_alignment_error": val_metrics.get("traj", 0.0),
                }
            )

            scheduler.step()

            elapsed = time.time() - start_t
            print(
                f"Epoch {epoch}: student train total={train_metrics['total']:.4f}, "
                f"val total={val_metrics['total']:.4f}, val acc={val_metrics['student_top1']:.2f}%, "
                f"val ce/kd/traj={val_metrics['ce']:.4f}/{val_metrics['kd']:.4f}/{val_metrics['traj']:.4f}, "
                f"time={elapsed:.1f}s"
            )

            self._save_student_checkpoint(last_ckpt, epoch, val_metrics["student_top1"], optimizer)
            if val_metrics["student_top1"] > best_val:
                best_val = val_metrics["student_top1"]
                self._save_student_checkpoint(best_ckpt, epoch, best_val, optimizer)
                print(f"[DirectTrajectoryZFlow] New best student checkpoint: {best_ckpt}")

        print(f"[DirectTrajectoryZFlow] Student stage complete. Best ckpt: {best_ckpt}")

    def _save_student_checkpoint(self, path: Path, epoch: int, metric: float, optimizer):
        payload = {
            "epoch": epoch,
            "metric": metric,
            "student_state_dict": self.student.state_dict() if self.student is not None else None,
            "student_adapters_state_dict": self.student_adapters.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "args": vars(self.args),
        }
        torch.save(payload, path)

    def _run_student_epoch(self, train: bool, optimizer):
        if self.student is None or self.student_hooks is None:
            raise RuntimeError("Student stage requires initialized student hooks")

        if self.args.traj_target_mode == "flow" and self.flow is None:
            raise RuntimeError("Flow must be initialized for flow target mode")

        self.student.train(mode=train)
        self.student_adapters.train(mode=train)

        if self.flow is not None:
            self.flow.eval()

        # Correctness checks: no gradients through frozen teacher and flow.
        if any(p.requires_grad for p in self.teacher.parameters()):
            raise RuntimeError("Teacher params must be frozen in student stage")
        if self.flow is not None and any(p.requires_grad for p in self.flow.parameters()):
            raise RuntimeError("Flow params must be frozen in student stage")

        meter_sums = defaultdict(float)
        sample_count = 0

        loader = self.train_loader if train else self.val_loader
        pbar = tqdm(loader, desc="Traj Student Train" if train else "Traj Student Val", leave=False)

        for batch_idx, batch in enumerate(pbar):
            inputs, targets = self._extract_inputs_targets(batch)
            bsz = inputs.size(0)

            with torch.no_grad():
                teacher_logits, teacher_hidden = self._teacher_forward(inputs)
                if self.args.traj_target_mode == "flow":
                    h_start = teacher_hidden[self.start_layer_1b - 1]
                    teacher_targets = generate_teacher_trajectory_targets(
                        h_start=h_start,
                        velocity_model=self.flow,
                        student_num_layers=len(self.student_block_modules),
                        flow_steps=self.args.traj_flow_steps,
                        start_t=0.0,
                        end_t=1.0,
                    )
                else:
                    teacher_targets = self._build_discrete_targets(
                        teacher_hidden, len(self.student_block_modules)
                    )
                if len(teacher_targets) != len(self.student_block_modules):
                    raise RuntimeError(
                        f"Target count mismatch: got {len(teacher_targets)} targets for {len(self.student_block_modules)} student layers"
                    )

            if train:
                optimizer.zero_grad()

            grad_ctx = torch.enable_grad() if train else torch.no_grad()
            with grad_ctx:
                student_logits, student_hidden = self._student_forward(inputs)

                traj_losses = []
                traj_detail = {}
                for k, (s_h, t_h) in enumerate(zip(student_hidden, teacher_targets), start=1):
                    key = f"layer_{k}"
                    adapter = self.student_adapters[key] if key in self.student_adapters else None
                    s_match, t_match = self._align_student_hidden(s_h, t_h, adapter)
                    l = F.mse_loss(s_match, t_match)
                    traj_losses.append(l)
                    traj_detail[f"traj_layer_{k}"] = l.item()

                if len(traj_losses) == 0:
                    traj_loss = student_logits.new_tensor(0.0)
                else:
                    traj_loss = torch.stack(traj_losses).mean()

                ce_loss = F.cross_entropy(student_logits, targets)
                kd_loss = self._kl_to_teacher(student_logits, teacher_logits)

                total_loss = (
                    self.args.alpha * ce_loss
                    + self.args.beta * kd_loss
                    + self.args.traj_lambda_hidden * traj_loss
                )

            if train:
                total_loss.backward()
                optimizer.step()

            student_top1 = accuracy(student_logits, targets, topk=(1,))[0].item()
            teacher_top1 = accuracy(teacher_logits, targets, topk=(1,))[0].item()
            kl_to_teacher = self._kl_to_teacher(student_logits, teacher_logits).item()

            metrics = {
                "total": total_loss.item(),
                "ce": ce_loss.item(),
                "kd": kd_loss.item(),
                "traj": traj_loss.item(),
                "student_top1": student_top1,
                "teacher_top1": teacher_top1,
                "kl_to_teacher": kl_to_teacher,
            }
            metrics.update(traj_detail)

            for key, value in metrics.items():
                meter_sums[key] += float(value) * bsz
            sample_count += bsz

            if batch_idx % max(1, self.args.print_freq // 10) == 0:
                pbar.set_postfix(
                    {
                        "total": f"{meter_sums['total']/max(sample_count,1):.4f}",
                        "ce": f"{meter_sums['ce']/max(sample_count,1):.4f}",
                        "traj": f"{meter_sums['traj']/max(sample_count,1):.4f}",
                        "acc": f"{meter_sums['student_top1']/max(sample_count,1):.2f}%",
                    }
                )

        return {k: v / max(sample_count, 1) for k, v in meter_sums.items()}
