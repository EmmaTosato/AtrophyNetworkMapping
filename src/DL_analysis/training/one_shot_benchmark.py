
import os
import sys
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import argparse

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from DL_analysis.cnn.datasets import FCDataset, AugmentedFCDataset
from DL_analysis.cnn.models import ResNet3D, VGG16_3D, AlexNet3D
from DL_analysis.training.train import train, validate
from DL_analysis.testing.test import evaluate

def load_metadata(group1, group2, metadata_path):
    df = pd.read_csv(metadata_path)
    df = df[df['Group'].isin([group1, group2])].reset_index(drop=True)
    return df

def get_model(model_name, n_classes=2, input_channels=1, device='cpu'):
    if model_name == 'resnet':
        return ResNet3D(n_classes=n_classes, in_channels=input_channels).to(device)
    elif model_name == 'alexnet':
        return AlexNet3D(num_classes=n_classes, input_channels=input_channels).to(device)
    elif model_name == 'vgg16':
        return VGG16_3D(num_classes=n_classes, input_channels=input_channels).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def one_shot_train(args):
    print(f"--- ONE SHOT TRAINING BENCHMARK ---")
    print(f"Group 1: {args.group1}")
    print(f"Group 2: {args.group2}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    df = load_metadata(args.group1, args.group2, args.metadata_path)
    print(f"Total Subjects: {len(df)}")
    
    # Simple Train/Test Split (80/20)
    X = df['ID'].values
    y = df['Group'].values
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Datasets
    # Train: Augmented
    train_dataset = AugmentedFCDataset(args.data_dir_augmented, train_df, 'Group', task='classification')
    # Test: Original
    test_dataset = FCDataset(args.data_dir, test_df, 'Group', task='classification')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = get_model(args.model, device=device)
    
    # Optimizer (SGD default)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = None
    if args.model == 'resnet':
        # Dynamic MultiStepLR
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.model in ['alexnet', 'vgg16']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    print("\nStarting Training...")
    start_time = time.time()
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step Scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

    total_time = time.time() - start_time
    print(f"\n--- TRAINING FINISHED ---")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Average Time per Epoch: {total_time/args.epochs:.2f}s")
    
    # Validate Final
    print("\nFinal Evaluation on Test Set:")
    y_true, y_pred = evaluate(model, val_loader, device)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred)
    print(f"Final Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group1", type=str, default="AD")
    parser.add_argument("--group2", type=str, default="PSP")
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--metadata_path", type=str, default="/data/users/etosato/ANM_Verona/assets/metadata/labels.csv")
    parser.add_argument("--data_dir", type=str, default="/data/users/etosato/ANM_Verona/data/FCmaps_processed")
    parser.add_argument("--data_dir_augmented", type=str, default="/data/users/etosato/ANM_Verona/data/FCmaps_augmented_processed")
    
    args = parser.parse_args()
    one_shot_train(args)
