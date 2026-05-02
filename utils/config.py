import argparse
from utils.data import get_available_domains, validate_domain_config, DOMAINBED_DATASETS


def str2bool(v):
    """Parse bool-like CLI values."""
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {v}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")

    # Model configuration
    parser.add_argument("--seed", type=int, default=42, help="Seed param")
    parser.add_argument(
        "--teacher", type=str, default="resnet50", help="Teacher architecture"
    )
    parser.add_argument(
        "--student", type=str, default="resnet18", help="Student architecture"
    )
    parser.add_argument(
        "--teacher_weights", type=str, help="Teacher pretrained weights path"
    )
    parser.add_argument(
        "--student_weights", type=str, help="Student pretrained weights path"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained weights for teacher model",
    )

    # Dataset configuration
    all_datasets = ["CIFAR100", "CIFAR10", "IMAGENETTE", "FOOD101", "PACS"] + list(
        DOMAINBED_DATASETS.keys()
    )
    parser.add_argument("--dataset", type=str, default="CIFAR100", choices=all_datasets)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--jitter", action="store_true", help="Apply color jitter")

    # Domain-based dataset arguments
    parser.add_argument(
        "--train_domains", type=str, nargs="+", help="Domains for training"
    )
    parser.add_argument(
        "--val_domains", type=str, nargs="+", help="Domains for validation"
    )
    parser.add_argument(
        "--classic_split",
        action="store_true",
        help="Use classic ML setup: train/val split from same domains",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Validation fraction in classic split",
    )
    parser.add_argument(
        "--test_env", type=int, default=None, help="[DomainBed only] Test domain ID"
    )

    # Method configuration
    parser.add_argument(
        "--method",
        type=str,
        default="LogitKD",
        choices=[
            "Pretraining",
            "HardCE",
            "LogitKD",
            "LogitMSE",
            "FlowKD",
            "ZFlow",
            "DKD",
            "DisDKD",
            "FitNet",
            "CRD",
            "ContraDKD",
        ],
    )
    parser.add_argument("--teacher_layer", type=str, default="layer3")
    parser.add_argument("--student_layer", type=str, default="layer2")
    parser.add_argument(
        "--adapter", type=str, default="student", choices=["student", "teacher"]
    )
    parser.add_argument(
        "--feat_dim", type=int, default=128, help="Feature dimension for projection"
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=128,
        help="Hidden channels for feature transformation",
    )

    # Method-specific hyperparameters
    parser.add_argument(
        "--disdkd_adversarial_weight",
        type=float,
        default=0.01,
        help="DisDKD adversarial weight (gamma)",
    )
    parser.add_argument(
        "--disc_lr_multiplier",
        type=float,
        default=1.0,
        help="Discriminator LR multiplier",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=1e-4,
        help="Discriminator learning rate",
    )
    parser.add_argument("--dkd_alpha", type=float, default=1.0, help="DKD TCKD weight")
    parser.add_argument("--dkd_beta", type=float, default=8.0, help="DKD NCKD weight")
    parser.add_argument(
        "--fitnet_stage1_epochs",
        type=int,
        default=0,
        help="Number of epochs for FitNet Stage 1 (Hint only)",
    )
    parser.add_argument(
        "--base_logits",
        type=str,
        default="zeros",
        choices=["zeros", "gaussian"],
        help="Base distribution for FlowKD interpolation",
    )
    parser.add_argument(
        "--lambda_fm",
        type=float,
        default=1.0,
        help="Weight for flow-matching loss in FlowKD",
    )
    parser.add_argument(
        "--use_kl",
        type=str2bool,
        default=True,
        help="Use KL distillation term for FlowKD",
    )
    parser.add_argument(
        "--flow_target",
        type=str,
        default="logits",
        choices=["logits", "probabilities"],
        help="Target space for FlowKD velocity matching",
    )
    parser.add_argument(
        "--debug_flowkd",
        type=str2bool,
        default=False,
        help="Print one-time FlowKD diagnostics on first batch",
    )
    parser.add_argument(
        "--zflow_stage",
        type=str,
        default="zspace",
        choices=["zspace", "flow", "eval", "all"],
        help="ZFlow stage to run",
    )
    parser.add_argument(
        "--z_layers",
        nargs="+",
        default=["layer1", "layer2", "layer3", "layer4"],
        help="Teacher layers used by ZFlow",
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=128,
        help="Common latent channel dimension for ZFlow",
    )
    parser.add_argument(
        "--z_tokens",
        type=int,
        default=4,
        help="Number of latent tokens for ZFlow adapters",
    )
    parser.add_argument(
        "--z_tokens_mode",
        type=str,
        default="spatial",
        choices=["global", "spatial"],
        help="CNN tokenization mode for ZFlow encoder",
    )
    parser.add_argument(
        "--z_use_norm",
        type=str2bool,
        default=True,
        help="Apply token normalization inside ZFlow adapters",
    )
    parser.add_argument(
        "--z_mlp_expansion",
        type=float,
        default=1.0,
        help="MLP expansion ratio for ZFlow encoder/decoder adapters",
    )
    parser.add_argument(
        "--lambda_rec",
        type=float,
        default=1.0,
        help="Stage-1 reconstruction loss weight",
    )
    parser.add_argument(
        "--lambda_dist",
        type=float,
        default=1.0,
        help="Stage-1 RKD distance loss weight",
    )
    parser.add_argument(
        "--lambda_angle",
        type=float,
        default=1.0,
        help="Stage-1 RKD angle loss weight",
    )
    parser.add_argument(
        "--lambda_trans_dist",
        type=float,
        default=1.0,
        help="Stage-1 transition RKD distance loss weight",
    )
    parser.add_argument(
        "--lambda_trans_angle",
        type=float,
        default=1.0,
        help="Stage-1 transition RKD angle loss weight",
    )
    parser.add_argument(
        "--lambda_path",
        type=float,
        default=1.0,
        help="Stage-2 path-consistency loss weight",
    )
    parser.add_argument(
        "--lambda_end",
        type=float,
        default=1.0,
        help="Stage-2 endpoint rollout loss weight",
    )
    parser.add_argument(
        "--zflow_pair_mode",
        type=str,
        default="mixed",
        choices=["adjacent", "long", "mixed"],
        help="Layer pair sampling mode for flow learning/eval",
    )
    parser.add_argument(
        "--zflow_num_pairs",
        type=int,
        default=4,
        help="Number of layer pairs sampled per batch in stage-2",
    )
    parser.add_argument(
        "--zflow_steps",
        type=int,
        default=8,
        help="Number of Euler steps used for stage-2 rollout/path losses",
    )
    parser.add_argument(
        "--zflow_eval_steps",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Euler step counts used for stage-3 flow replacement evaluation",
    )
    parser.add_argument(
        "--zflow_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of ZFlow velocity network",
    )
    parser.add_argument(
        "--zflow_num_blocks",
        type=int,
        default=2,
        help="Number of residual blocks in ZFlow velocity network",
    )
    parser.add_argument(
        "--zflow_diagnostic",
        type=str2bool,
        default=True,
        help="Enable oracle diagnostic baselines/metrics in stage-3",
    )
    parser.add_argument(
        "--zspace_ckpt",
        type=str,
        default="",
        help="Path to Stage-1 Z-space checkpoint",
    )
    parser.add_argument(
        "--flow_ckpt",
        type=str,
        default="",
        help="Path to Stage-2 flow checkpoint",
    )

    # ContraDKD-specific hyperparamters
    parser.add_argument(
        "--l2c_weight",
        type=float,
        default=0.5,
        help="Weight for L2C contrastive loss in ContraDKD",
    )

    # CRD-specific hyperparameters
    parser.add_argument(
        "--crd_temperature",
        type=float,
        default=0.07,
        help="Temperature for CRD contrastive loss",
    )
    parser.add_argument(
        "--crd_momentum",
        type=float,
        default=0.5,
        help="Momentum for CRD memory bank updates",
    )
    parser.add_argument(
        "--crd_n_negatives",
        type=int,
        default=4096,
        help="Number of negative samples for CRD",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd", "adamw"]
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument(
        "--step_size", type=int, default=5, help="LR scheduler step size"
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.8,
        help="LR decay factor after every step-size",
    )

    # Loss weights
    parser.add_argument("--alpha", type=float, default=1.0, help="CE loss weight")
    parser.add_argument("--beta", type=float, default=0.4, help="KD loss weight")
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Method-specific loss weight"
    )
    parser.add_argument("--tau", type=float, default=4.0, help="Temperature for KD")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Alias for --tau (for experiment compatibility)",
    )

    # Logging and output
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_file", type=str, default="log.csv")

    args = parser.parse_args()
    if args.temperature is not None:
        args.tau = args.temperature
    else:
        args.temperature = args.tau
    return args


def validate_and_setup_domains(args):
    """Validate and setup domain configuration for domain-based datasets."""
    dataset_name = args.dataset.upper()

    if dataset_name not in DOMAINBED_DATASETS and dataset_name not in [
        "PACS",
        "PACS_DEEPLAKE",
    ]:
        if args.train_domains or args.val_domains or args.classic_split:
            print(
                f"Warning: Domain arguments ignored for non-domain-based dataset {args.dataset}"
            )
        args.train_domains = None
        args.val_domains = None
        args.classic_split = False
        return

    available_domains = get_available_domains(args.dataset)

    if args.train_domains is None:
        if args.classic_split:
            args.train_domains = available_domains
            print(f"Classic split: using all domains {args.train_domains}")
        else:
            args.train_domains = available_domains[:-1]
            if args.val_domains is None:
                args.val_domains = [available_domains[-1]]
            print(
                f"OOD split - Training: {args.train_domains}, Validation: {args.val_domains}"
            )

    if args.classic_split:
        if args.val_domains is not None:
            print("Warning: --val_domains ignored in classic split mode.")
        args.val_domains = None
        print(f"Classic ML setup with domains: {args.train_domains}")
        print(
            f"Train/val split: {int((1-args.test_size)*100)}%/{int(args.test_size*100)}%"
        )
    else:
        if args.val_domains is None:
            args.val_domains = [
                d for d in available_domains if d not in args.train_domains
            ]
            if not args.val_domains:
                raise ValueError(f"No available validation domains.")
            print(f"Using remaining domains for validation: {args.val_domains}")

    validate_domain_config(
        args.dataset, args.train_domains, args.val_domains, args.classic_split
    )


def print_training_config(args):
    """Print complete training configuration."""
    print(f"\n=== Training Configuration ===")
    print(f"Method: {args.method}")
    print(f"Teacher: {args.teacher} -> Student: {args.student}")
    print(f"Dataset: {args.dataset}")

    dataset_name = args.dataset.upper()
    if dataset_name in DOMAINBED_DATASETS or dataset_name in ["PACS", "PACS_DEEPLAKE"]:
        if args.classic_split:
            print(f"Setup: Classic ML (train/val split from same domains)")
            print(f"Domains: {args.train_domains}")
            print(
                f"Train/Val split: {int((1-args.test_size)*100)}%/{int(args.test_size*100)}%"
            )
        else:
            print(f"Setup: Out-of-Distribution evaluation")
            print(f"Training domains: {args.train_domains}")
            print(f"Validation domains: {args.val_domains}")

    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}")
    print(f"Loss weights - α: {args.alpha}, β: {args.beta}, γ: {args.gamma}")
    print(f"Temperature: {args.tau}")

    # Method-specific configurations
    if args.method == "DKD":
        print(f"DKD weights - TCKD α: {args.dkd_alpha}, NCKD β: {args.dkd_beta}")

    elif args.method == "DisDKD":
        print(f"DisDKD DKD weights - TCKD α: {args.dkd_alpha}, NCKD β: {args.dkd_beta}")
        print(f"DisDKD Adversarial weight (γ): {args.disdkd_adversarial_weight}")
        print(
            f"Discriminator LR: {args.discriminator_lr}, multiplier: {args.disc_lr_multiplier}"
        )

    elif args.method == "CRD":
        print(f"CRD Temperature: {args.crd_temperature}")
        print(f"CRD Momentum: {args.crd_momentum}")
        print(f"CRD Negative samples: {args.crd_n_negatives}")
        print(f"CRD Feature dimension: {args.feat_dim}")
        print(
            f"CRD Teacher layer: {args.teacher_layer}, Student layer: {args.student_layer}"
        )

    elif args.method == "FitNet":
        print(
            f"FitNet Teacher layer: {args.teacher_layer}, Student layer: {args.student_layer}"
        )
        if args.fitnet_stage1_epochs > 0:
            print(f"FitNet Stage 1 epochs (Hint only): {args.fitnet_stage1_epochs}")
            print(
                f"FitNet Stage 2 epochs (Task): {args.epochs - args.fitnet_stage1_epochs}"
            )
    elif args.method == "LogitMSE":
        print(f"LogitMSE weight (γ): {args.gamma}")
    elif args.method == "FlowKD":
        print(f"FlowKD base logits: {args.base_logits}")
        print(f"FlowKD lambda_fm: {args.lambda_fm}")
        print(f"FlowKD target space: {args.flow_target}")
        print(f"FlowKD use KL: {args.use_kl}, KL weight (β): {args.beta}")
        print(f"FlowKD temperature: {args.tau}")
        print(f"FlowKD debug diagnostics: {args.debug_flowkd}")
    elif args.method == "ZFlow":
        print(f"ZFlow stage: {args.zflow_stage}")
        print(f"ZFlow layers: {args.z_layers}")
        print(
            f"ZFlow latent config: z_dim={args.z_dim}, z_tokens={args.z_tokens}, "
            f"z_tokens_mode={args.z_tokens_mode}, use_norm={args.z_use_norm}"
        )
        print(
            f"ZFlow Stage-1 lambdas: rec={args.lambda_rec}, dist={args.lambda_dist}, "
            f"angle={args.lambda_angle}, trans_dist={args.lambda_trans_dist}, "
            f"trans_angle={args.lambda_trans_angle}"
        )
        print(
            f"ZFlow Stage-2: lambda_fm={args.lambda_fm}, lambda_path={args.lambda_path}, "
            f"lambda_end={args.lambda_end}, pair_mode={args.zflow_pair_mode}, "
            f"num_pairs={args.zflow_num_pairs}, steps={args.zflow_steps}"
        )
        print(f"ZFlow Eval steps: {args.zflow_eval_steps}")
        print(f"ZFlow diagnostics enabled: {args.zflow_diagnostic}")
        if args.zspace_ckpt:
            print(f"ZFlow zspace_ckpt: {args.zspace_ckpt}")
        if args.flow_ckpt:
            print(f"ZFlow flow_ckpt: {args.flow_ckpt}")

    print("=" * 40)
