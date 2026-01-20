
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from itertools import product
from copy import deepcopy

# Add project root to path
# We need to add 'src' to sys.path so that 'from DL_analysis...' imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
# src/DL_analysis/training -> ../../ -> src
src_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from DL_analysis.cnn.datasets import FCDataset, AugmentedFCDataset
from DL_analysis.cnn.models import ResNet3D, DenseNet3D, VGG16_3D
try:
    from DL_analysis.cnn.models import AlexNet3D
except ImportError:
    AlexNet3D = None  # To be implemented

from DL_analysis.training.train import train, validate
from DL_analysis.testing.test import evaluate, compute_metrics

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_metadata(group1, group2, metadata_path):
    """Load and filter metadata for the two groups."""
    df = pd.read_csv(metadata_path)
    # Filter groups
    df = df[df['Group'].isin([group1, group2])].reset_index(drop=True)
    return df

def get_model(model_name, n_classes=2, input_channels=1, device='cpu'):
    if model_name == 'resnet':
        return ResNet3D(n_classes=n_classes, in_channels=input_channels).to(device)
    elif model_name == 'alexnet':
        if AlexNet3D is None:
            raise NotImplementedError("AlexNet3D not implemented yet")
        return AlexNet3D(num_classes=n_classes, input_channels=input_channels).to(device)
    elif model_name == 'vgg16':
        return VGG16_3D(num_classes=n_classes, input_channels=input_channels).to(device)
    elif model_name == 'densenet':
        return DenseNet3D(n_classes=n_classes).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_single_fold(train_dataset, val_dataset, config, device, verbose=False):
    """Train a model for one fold (used in inner CV and full retrain)."""
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    model = get_model(config['model_type'], device=device)
    
    # Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
        
    # Training Loop
    best_acc = -1
    best_model_state = None
    patience = config.get('patience', 10)
    patience_counter = 0
    
    epochs = config['epochs']
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            if verbose: print("  Early stopping triggered.")
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_acc

def inner_cv_grid_search(train_df, model_name, grid_params, data_dirs, seed, device):
    """
    Perform Inner CV (5-fold) to find best hyperparameters.
    train_df: DataFrame with training subjects for this outer fold.
    """
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    # Generate all combinations
    keys = list(grid_params.keys())
    combinations = list(product(*grid_params.values()))
    
    best_score = -1
    best_config = None
    
    print(f"  > Starting Grid Search ({len(combinations)} configs)...")
    
    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        config['model_type'] = model_name
        # Defaults
        config['optimizer'] = 'sgd' # Default to SGD as per literature? Or parameterize
        config['epochs'] = 60 # Set a reasonable default or pass via args
        if 'epochs' in grid_params: config['epochs'] = grid_params['epochs'] # Allow override if in grid

        fold_scores = []
        
        for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(train_df['ID'], train_df['Group'])):
            # Inner Split
            inner_train_df = train_df.iloc[train_idx]
            inner_val_df = train_df.iloc[val_idx]
            
            # Datasets: Train is AUGMENTED, Val is ORIGINAL
            train_dataset = AugmentedFCDataset(data_dirs['augmented'], inner_train_df, 'Group', task='classification')
            val_dataset = FCDataset(data_dirs['original'], inner_val_df, 'Group', task='classification')
            
            # Train on inner fold
            _, acc = train_single_fold(train_dataset, val_dataset, config, device, verbose=False)
            fold_scores.append(acc)
            
        mean_score = np.mean(fold_scores)
        # print(f"    Config {i+1}: {config} -> Mean Acc: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_config = config
            
    print(f"  > Best Config: {best_config} (Acc: {best_score:.4f})")
    return best_config

def nested_cv_classification(args):
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    data_dirs = {
        'original': args.data_dir,
        'augmented': args.data_dir_augmented
    }
    
    # Load Metadata
    print(f"Loading metadata for {args.group1} vs {args.group2}...")
    df_meta = load_metadata(args.group1, args.group2, args.metadata_path)
    X = df_meta['ID'].values
    y = df_meta['Group'].values
    print(f"Total subjects: {len(df_meta)}")
    
    # Output Dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Outer CV
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    # Load Grid Params
    with open(args.config_path, 'r') as f:
        full_config = json.load(f)
        grid_params = full_config['grids'][args.model]
        # Add common fixed params if needed
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        if args.test_mode and outer_fold > 0: break # Only 1 fold in test mode
        
        print(f"\n=== OUTER FOLD {outer_fold+1}/5 ===")
        
        # Split
        train_df = df_meta.iloc[train_idx].reset_index(drop=True)
        test_df = df_meta.iloc[test_idx].reset_index(drop=True)
        
        if args.test_mode:
            print("Running in TEST MODE (reduced dataset)")
            train_df = train_df.iloc[:20]
            test_df = test_df.iloc[:5]
        
        # 1. Inner CV Grid Search
        print("Step 1: Inner CV Grid Search")
        best_params = inner_cv_grid_search(train_df, args.model, grid_params, data_dirs, seed=42, device=device)
        
        # 2. Full Retrain
        print("Step 2: Full Retrain with Best Params")
        
        from sklearn.model_selection import train_test_split
        # Using a small validation split for early stopping
        retrain_train_df, retrain_val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['Group'], random_state=42)
        
        train_dataset = AugmentedFCDataset(data_dirs['augmented'], retrain_train_df, 'Group', task='classification')
        val_dataset = FCDataset(data_dirs['original'], retrain_val_df, 'Group', task='classification')
        
        final_model, _ = train_single_fold(train_dataset, val_dataset, best_params, device, verbose=True)
        
        # 3. Test
        print("Step 3: Final Testing")
        test_dataset = FCDataset(data_dirs['original'], test_df, 'Group', task='classification')
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        y_true, y_pred = evaluate(final_model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        
        # Save results
        fold_result = {
            'fold': outer_fold + 1,
            'best_params': best_params,
            'metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k,v in metrics.items() if k!='confusion_matrix'}
        }
        results.append(fold_result)
        
        # Save Model
        fold_dir = os.path.join(args.output_dir, f"fold_{outer_fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(final_model.state_dict(), os.path.join(fold_dir, "best_model.pt"))
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(fold_result, f, indent=4)
            
        print(f"Fold {outer_fold+1} Accuracy: {metrics['accuracy']:.4f}")

    # Aggregate
    print("\n=== AGGREGATED RESULTS ===")
    agg_metrics = {}
    metric_keys = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    for k in metric_keys:
        values = [r['metrics'][k] for r in results]
        agg_metrics[f"mean_{k}"] = np.mean(values)
        agg_metrics[f"std_{k}"] = np.std(values)
        
    print(json.dumps(agg_metrics, indent=4))
    
    with open(os.path.join(args.output_dir, "aggregated_results.json"), "w") as f:
        json.dump({'aggregate': agg_metrics, 'folds': results}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group1", type=str, required=True)
    parser.add_argument("--group2", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="src/DL_analysis/config/nested_cv_config.json")
    parser.add_argument("--metadata_path", type=str, default="/data/users/etosato/ANM_Verona/assets/metadata/labels.csv")
    parser.add_argument("--data_dir", type=str, default="/data/users/etosato/ANM_Verona/data/FCmaps_processed")
    parser.add_argument("--data_dir_augmented", type=str, default="/data/users/etosato/ANM_Verona/data/FCmaps_augmented_processed")
    parser.add_argument("--output_dir", type=str, default="results/nested_cv_output")
    parser.add_argument("--test_mode", action='store_true')
    
    args = parser.parse_args()
    
    nested_cv_classification(args)
