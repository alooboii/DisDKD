# utils/tsne.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from tqdm import tqdm

# Constants for Checkpoint Names
TEACHER_CHECKPOINT_TEMPLATE = "teacher_epoch_{:03d}.pth"
STUDENT_CHECKPOINT_TEMPLATE = "student_epoch_{:03d}.pth"

def extract_features(model, dataloader, device):
    """
    Extracts features (embeddings before the final classifier layer) and labels.
    Assumes the model has a 'feature_extractor' attribute, or that the last 
    module before the classifier is the one we want to extract features from.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # We will try to extract features from the 'feature_extractor' or the layer before 'fc'
    
    # Determine the feature extraction module based on common conventions
    # If the model has a 'feature_extractor' property (e.g., resnet.avgpool)
    if hasattr(model, 'feature_extractor'):
        feature_module = model.feature_extractor
    # If it's a standard model structure, let's use the layer before the final FC layer
    elif hasattr(model, 'fc'): # assuming a standard ResNet-like structure
        # Temporarily detach the final fully connected layer
        temp_fc = model.fc 
        model.fc = torch.nn.Identity() # Replace with identity to get pre-classification features
        feature_module = model
    else:
        # Fallback to the whole model if we can't easily identify the feature layer
        print("Warning: Could not identify feature extractor. Using output before final classifier.")
        feature_module = model

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            # labels = labels.to(device) # Keep labels on CPU for numpy/plotting

            if feature_module is model:
                # If we replaced model.fc with Identity:
                outputs = model(inputs)
                features = outputs.view(outputs.size(0), -1) # Flatten features
            elif hasattr(model, 'feature_extractor'):
                 # If model has a dedicated feature_extractor property
                features = feature_module(inputs).view(inputs.size(0), -1)
            else:
                # If we can't find the feature_extractor or fc to disable, 
                # this is a guess that the output before classifier is what we want.
                # This part is highly dependent on your actual model architecture (utils/models.py)
                features = model(inputs) 
                features = features.view(features.size(0), -1)


            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    # Revert model change if we modified it
    if hasattr(model, 'fc') and feature_module is not model:
        model.fc = temp_fc

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels


def generate_tsne_plot(args, model, dataloader, device):
    """
    Generates and saves the t-SNE plot for the given model and data.
    """
    print(f"\n--- Starting t-SNE Generation ---")
    print(f"Data Split: {args.tsne_data_split.upper()}, Samples: {args.tsne_n_samples}")
    
    # 1. Feature Extraction
    all_features, all_labels = extract_features(model, dataloader, device)
    
    # 2. Sampling (if needed)
    if all_features.shape[0] > args.tsne_n_samples:
        print(f"Sampling {args.tsne_n_samples} samples...")
        indices = np.random.choice(all_features.shape[0], args.tsne_n_samples, replace=False)
        features = all_features[indices]
        labels = all_labels[indices]
    else:
        features = all_features
        labels = all_labels
        
    print(f"Features shape: {features.shape}")
    
    # 3. Run t-SNE 
    print("Running t-SNE (this might take a few moments)...")
    tsne = TSNE(n_components=2, random_state=args.seed, n_jobs=-1)
    # Common parameters for t-SNE: perplexity=30, learning_rate=200, n_iter=1000
    features_2d = tsne.fit_transform(features)
    
    # 4. Plotting
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Generate a colormap
    cmap = plt.cm.get_cmap('viridis', num_classes)
    
    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(
            features_2d[indices, 0],
            features_2d[indices, 1],
            label=f"Class {label}",
            color=cmap(i),
            alpha=0.6,
            s=10 # point size
        )

    plt.title(
        f"t-SNE of {args.tsne_model.capitalize()} Features (Epoch {args.tsne_epoch})\n"
        f"Dataset: {args.dataset}, Split: {args.tsne_data_split.upper()}"
    )
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to fit legend
    
    # 5. Save the plot
    save_path = os.path.join(args.save_dir, args.tsne_output_file)
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"t-SNE plot successfully saved to: {save_path}")
    

def run_tsne_visualization(args):
    """
    Main function to orchestrate the loading and plotting.
    """
    if args.tsne_epoch is None:
        raise ValueError(
            "The --tsne_epoch flag is required when --tsne_plot is enabled."
        )

    # Setup device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    
    # --- 1. Determine Checkpoint Path ---
    if args.tsne_model == "teacher":
        filename = TEACHER_CHECKPOINT_TEMPLATE.format(args.tsne_epoch)
    else: # student
        filename = STUDENT_CHECKPOINT_TEMPLATE.format(args.tsne_epoch)
        
    checkpoint_path = os.path.join(args.save_dir, filename)
    print(f"Attempting to load checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # --- 2. Load Data Loaders ---
    # We must import get_dataloaders and other utilities from your system
    from utils.data import get_dataloaders
    from utils.models import TeacherModel, StudentModel
    from utils.utils import load_checkpoint
    from utils.config import validate_and_setup_domains
    
    if args.dataset.upper() in [
        "VLCS",
        "PACS",
        "PACS_DEEPLAKE",
        "OFFICEHOME",
        "OFFICE_HOME",
    ]:
         # Re-run domain setup needed for correct data loading
        validate_and_setup_domains(args)
    
    # Load dataset to determine num_classes (temporarily re-using the data loading logic)
    train_loader, val_loader, num_classes = get_dataloaders(
        args.dataset,
        args.batch_size,
        args.data_root,
        args.jitter,
        args.num_workers,
        args.train_domains,
        args.val_domains,
        args.classic_split,
        args.test_size,
    )
    
    # Select the data loader based on the flag
    if args.tsne_data_split == "val":
        dataloader_to_use = val_loader
    else:
        dataloader_to_use = train_loader


    # --- 3. Instantiate and Load Model ---
    if args.tsne_model == "teacher":
        model = TeacherModel(
            args.teacher, num_classes, args.teacher_weights, pretrained=args.pretrained
        ).to(device)
    else: # student
        model = StudentModel(
            args.student, num_classes, args.student_weights
        ).to(device)

    # Load weights
    load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f"Successfully loaded {args.tsne_model.capitalize()} model from epoch {args.tsne_epoch}")

    # --- 4. Generate Plot ---
    generate_tsne_plot(args, model, dataloader_to_use, device)
    
    print("\n--- t-SNE Visualization Complete ---")