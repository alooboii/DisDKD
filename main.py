import argparse
from pathlib import Path
import torch
import torch.nn as nn

from utils.data import get_dataloaders, get_crd_dataloaders, print_dataset_info
from utils.models import TeacherModel, StudentModel
from utils.utils import set_seed
from utils.config import parse_args, validate_and_setup_domains, print_training_config
from utils.training import Trainer
from utils.logging import LossTracker


def main():
    args = parse_args()
    print(f'Using seed ===> {args.seed}')
    set_seed(args.seed)
    
    # Validate domain configuration for domain-based datasets
    if args.dataset.upper() in ['VLCS', 'PACS', 'PACS_DEEPLAKE', 'OFFICEHOME', 'OFFICE_HOME']:
        if args.train_domains is None:
            args.train_domains = []
        validate_and_setup_domains(args)
    
    # Setup device and directories
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    
    # Print dataset information
    print_dataset_info(args.dataset, args.train_domains, args.val_domains, args.classic_split)
    
    # Load datasets
    if args.method == "CRD":
        train_loader, val_loader, num_classes = get_crd_dataloaders(
            args.dataset, args.batch_size, args.data_root, args.jitter, 
            args.num_workers, args.train_domains, args.val_domains, 
            args.classic_split, args.test_size)
    else:
        train_loader, val_loader, num_classes = get_dataloaders(
            args.dataset, args.batch_size, args.data_root, args.jitter, 
            args.num_workers, args.train_domains, args.val_domains,
            args.classic_split, args.test_size)
    
    args._train_dataset = train_loader.dataset
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Print training configuration
    print_training_config(args)
    
    # Create models
    if args.method == 'Pretraining':
        teacher = TeacherModel(args.teacher, num_classes, args.teacher_weights, 
                              pretrained=args.pretrained).to(device)
        student = None
    else:
        teacher = TeacherModel(args.teacher, num_classes, args.teacher_weights).to(device)
        student = StudentModel(args.student, num_classes, args.student_weights).to(device)
    
    # Initialize trainer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    log_path = Path(args.save_dir) / args.log_file
    loss_tracker = LossTracker(log_path, args.method)
    
    trainer = Trainer(
        teacher=teacher,
        student=student,
        num_classes=num_classes,
        criterion=criterion,
        loss_tracker=loss_tracker,
        device=device,
        args=args
    )
    
    # Train the model
    best_acc = trainer.train(train_loader, val_loader)
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")
    print(f"Logs saved to: {log_path}")
    
    # Print final domain configuration summary
    if args.dataset.upper() in ['VLCS', 'PACS', 'PACS_DEEPLAKE', 'OFFICEHOME', 'OFFICE_HOME']:
        print(f"\nDomain Configuration Summary:")
        print(f"  Dataset: {args.dataset}")
        if args.classic_split:
            print(f"  Setup: Classic ML (train/val split from same domains)")
            print(f"  Domains used: {args.train_domains}")
        else:
            print(f"  Setup: Out-of-Distribution evaluation")
            print(f"  Training domains: {args.train_domains}")
            print(f"  Validation domains: {args.val_domains}")


if __name__ == '__main__':
    main()