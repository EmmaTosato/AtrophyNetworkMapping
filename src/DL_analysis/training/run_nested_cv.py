
import os
import sys
import json
import argparse
import re
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
from DL_analysis.cnn.models import ResNet3D, VGG16_3D
try:
    from DL_analysis.cnn.models import AlexNet3D
except ImportError:
    AlexNet3D = None  # To be implemented

from DL_analysis.training.train import train, validate, plot_training_curves
from DL_analysis.testing.test import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def compute_metrics_safe(y_true, y_pred, probas=None):
    """
    Compute metrics avoiding confusion matrix if not needed or if it causes issues.
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    # Use zero_division=0 to silence warnings
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC if probabilities are provided
    if probas is not None:
        try:
            # Handle binary case specifically if n_classes=2
            if probas.shape[1] == 2:
                # Use probability of positive class (index 1)
                metrics['auc'] = roc_auc_score(y_true, probas[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_true, probas, multi_class='ovr')
        except Exception as e:
            print(f"Warning: AUC calculation failed: {e}")
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics

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
    momentum = config.get('momentum', 0.9) # Default to 0.9 if missing (backward compat)
    
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
        
    # Scheduler
    scheduler = None
    gamma = config.get('scheduler_gamma', 0.1)
    sch_patience = config.get('scheduler_patience', 5)
    
    if config['model_type'] == 'resnet':
        # ResNet Paper: Divide by 10 at 50% and 75% of training
        total_epochs = config['epochs']
        milestones = [int(total_epochs * 0.5), int(total_epochs * 0.75)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif config['model_type'] in ['alexnet', 'vgg16']:
        # AlexNet/VGG Paper: Divide by 10 when validation error plateaus.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=gamma, patience=sch_patience)
        
    # Training Loop
    best_acc = -1
    best_model_state = None
    patience = config.get('patience', None) # Default to None (No Early Stopping)
    patience_counter = 0
    
    epochs = config['epochs']
    
    # History Tracking
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Track History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Step Scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc) # Monitor Val Accuracy directly
            else:
                scheduler.step()
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
        if val_acc > best_acc:
            best_acc = val_acc
            best_train_acc = train_acc # Keep track of train acc at best val point
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience and patience_counter >= patience:
            if verbose: print(f"  Early stopping at epoch {epoch+1}")
            break
            
    # Capture Last State (Checkpointing)
    last_model_state = deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # If using Test Mode with 1 epoch, best_train_acc might not be set if loop runs once and val_acc > -1
    # Ensure it's defined
    if 'best_train_acc' not in locals():
        best_train_acc = locals().get('train_acc', 0.0)

    return model, best_acc, best_train_acc, history, last_model_state

def inner_cv_grid_search(train_df, model_name, grid_params, data_dirs, seed, device, args):
    """
    Perform Inner CV (5-fold) to find best hyperparameters.
    train_df: DataFrame with training subjects for this outer fold.
    """
    # Test Mode overrides
    if args.test_mode:
         print("Overriding config for TEST MODE: epochs=1, inner_splits=2")
         inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    else:
         inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    # Generate all combinations
    keys = list(grid_params.keys())
    combinations = list(product(*grid_params.values()))
    
    best_score = -1
    best_config = None
    
    print(f"\n  [Inner Grid Search] Starting Grid Search ({len(combinations)} configurations)...")
    
    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        if args.test_mode: config['epochs'] = 1
        config['model_type'] = model_name
        # Defaults
        config['optimizer'] = 'sgd' # Default to SGD as per literature? Or parameterize
        
        # Handle 'epochs': grid_params['epochs'] is a list like [60]. 
        # If it's in the grid (which it is), 'config' already has it as a scalar from 'combo'.
        # However, if we defaults it, ensure it's scalar.
        if 'epochs' not in config:
             config['epochs'] = 60 
             
        # Handle 'patience' if in grid/config
        if 'patience' not in config:
             config['patience'] = None # Default
        
        # Double check types
        if isinstance(config.get('batch_size'), list): config['batch_size'] = config['batch_size'][0]
        if isinstance(config.get('epochs'), list): config['epochs'] = config['epochs'][0]
        if isinstance(config.get('patience'), list): config['patience'] = config['patience'][0]

        fold_scores = []
        
        fold_scores = []
        
        for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(train_df['ID'], train_df['Group'])):
            # Inner Split
            inner_train_df = train_df.iloc[train_idx]
            inner_val_df = train_df.iloc[val_idx]
            
            # Datasets: Train is AUGMENTED, Val is ORIGINAL
            train_dataset = AugmentedFCDataset(data_dirs['augmented'], inner_train_df, 'Group', task='classification')
            val_dataset = FCDataset(data_dirs['original'], inner_val_df, 'Group', task='classification')
            
            # Train on inner fold
            _, val_acc, train_acc, _, _ = train_single_fold(train_dataset, val_dataset, config, device, verbose=False)
            fold_scores.append(val_acc)
            
        mean_score = np.mean(fold_scores)
        print(f"    [Config {i+1}/{len(combinations)}] Mean Acc: {mean_score:.4f} | Params: {config}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_config = config
            
    print(f"  [Inner Grid Search] Best Config found (Acc: {best_score:.4f}):\n    {best_config}")
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
    
    # Redirect stdout to log file
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "run.log")
    
    # Save Full Config for Reproducibility
    # Config snapshot disabled as per user request (redundant with logs/global config)
    # with open(os.path.join(args.output_dir, "config_snapshot.json"), "w") as f:
    #     # Load grid used
    #     with open(args.config_path, 'r') as cf:
    #         original_config = json.load(cf)
    #     
    #     snapshot = {
    #         'args': vars(args),
    #         'original_config': original_config,
    #         'device': str(device)
    #     }
    #     
    #     # Determine strict formatting
    #     # We want to compact lists like [0.01] into single lines
    #     json_str = json.dumps(snapshot, indent=4)
    #     
    #     # Regex to find lists that are expanded and collapse them
    #     # Logic: Find [ followed by non-bracket content followed by ]
    #     # This safely collapses lists of numbers/strings/nulls
    #     json_str = re.sub(r'\[\s+([^\[\]\{\}]+?)\s+\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', json_str)
    #     
    #     f.write(json_str)
    print(f"Logging to {log_file}")
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout
    # Enable line buffering
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    
    print(f"Command arguments:")
    for arg, value in vars(args).items():
        print(f"  - {arg}: {value}")
    
    print(f"Using device: {device}")
    
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
            print("  [Outer Fold] Running in TEST MODE (reduced dataset)")
            train_df = train_df.iloc[:20]
            test_df = test_df.iloc[:5]
        
        print(f"  [Outer Fold] Train Size: {len(train_df)} subjects")
        print(f"  [Outer Fold] Test Size : {len(test_df)} subjects")
        
        # 1. Inner CV Grid Search
        print(f"\n--- Step 1: Inner CV Grid Search (Fold {outer_fold+1}) ---")
        best_params = inner_cv_grid_search(train_df, args.model, grid_params, data_dirs, seed=42, device=device, args=args)
        
        # 2. Full Retrain
        print(f"\n--- Step 2: Full Retrain with Best Params (Fold {outer_fold+1}) ---")
        
        if args.test_mode:
            best_params['epochs'] = 1
        
        from sklearn.model_selection import train_test_split
        # Using a small validation split for early stopping
        # Note: We retrain on the vast majority of the outer train set (90%), using 10% just for early stopping check
        retrain_train_df, retrain_val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['Group'], random_state=42)
        
        print(f"  [Retrain] Training on {len(retrain_train_df)} subjects (Augmented)")
        print(f"  [Retrain] Validation monitor on {len(retrain_val_df)} subjects (Original)")
        
        train_dataset = AugmentedFCDataset(data_dirs['augmented'], retrain_train_df, 'Group', task='classification')
        val_dataset = FCDataset(data_dirs['original'], retrain_val_df, 'Group', task='classification')
        
        final_model, final_val_acc, final_train_acc, history, last_model_state = train_single_fold(train_dataset, val_dataset, best_params, device, verbose=True)
        print(f"  [Retrain] Completed. Best Val Acc: {final_val_acc:.4f} | Train Acc at that point: {final_train_acc:.4f}")
        
        # 3. Test
        print(f"\n--- Step 3: Final Testing (Fold {outer_fold+1}) ---")
        print(f"  [Test] Evaluating on {len(test_df)} held-out subjects...")
        test_dataset = FCDataset(data_dirs['original'], test_df, 'Group', task='classification')
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        y_true, y_pred, y_probs = evaluate(final_model, test_loader, device)
        metrics = compute_metrics_safe(y_true, y_pred, y_probs)
        
        # Save results
        fold_result = {
            'fold': outer_fold + 1,
            'best_params': best_params,
            'metrics': metrics
        }
        results.append(fold_result)
        
        # --- SAVING ARTIFACTS ---
        fold_dir = os.path.join(args.output_dir, f"fold_{outer_fold+1}")
        plots_dir = os.path.join(fold_dir, "plots")
        models_dir = os.path.join(fold_dir, "models")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # 1. Plots
        plot_training_curves(history, plots_dir)
        
        # 2. Models (Best & Last)
        torch.save(final_model.state_dict(), os.path.join(models_dir, "best_model.pt"))
        torch.save(last_model_state, os.path.join(models_dir, "last_model.pt"))
        
        # 3. Validated Metrics
        # 3. Validated Metrics (Saved as Params only, as requested)
        # We save only best_params and fold info to metrics.json
        fold_info_only = {
             'fold': outer_fold + 1,
             'best_params': best_params
        }
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(fold_info_only, f, indent=4)
        
        # 4. History (Row-oriented for readability)
        history_list = []
        num_epochs = len(history['train_loss'])
        for i in range(num_epochs):
            entry = {'epoch': i + 1}
            for k, v in history.items():
                entry[k] = round(v[i], 4) # Round for readability
            history_list.append(entry)

        with open(os.path.join(fold_dir, "history.json"), "w") as f:
            json.dump(history_list, f, indent=4)
            
        print(f"  [Test] Fold {outer_fold+1} Completed. Accuracy: {metrics['accuracy']:.4f}")

    # Aggregate
    print("\n==================================")
    print("=== AGGREGATED NESTED CV RESULTS ===")
    print("==================================")
    agg_metrics = {}
    metric_keys = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    for k in metric_keys:
        values = [r['metrics'].get(k, 0.0) for r in results]
        agg_metrics[f"mean_{k}"] = np.mean(values)
        agg_metrics[f"std_{k}"] = np.std(values)
        
    print(json.dumps(agg_metrics, indent=4))
    
    # --- Save to CSV (Requested format) ---
    
    # 1. aggregated_results.csv (Mean/Std of metrics)
    # Round to 3 decimals
    agg_metrics_rounded = {k: round(v, 3) for k, v in agg_metrics.items()}
    df_agg = pd.DataFrame([agg_metrics_rounded])
    df_agg.to_csv(os.path.join(args.output_dir, "aggregated_results.csv"), index=False)
    
    # 2. nested_cv_results.csv (Per-fold metrics)
    nested_rows = []
    for r in results:
        row = {'fold': r['fold']}
        # Flatten metrics
        for k, v in r['metrics'].items():
            row[k] = round(v, 3) if isinstance(v, (int, float)) else v
        nested_rows.append(row)
        
    df_nested = pd.DataFrame(nested_rows)
    df_nested.to_csv(os.path.join(args.output_dir, "nested_cv_results.csv"), index=False)
    
    print(f"Results saved to {args.output_dir}/aggregated_results.csv and nested_cv_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group1", type=str, required=True)
    parser.add_argument("--group2", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="src/DL_analysis/config/cnn_grid.json")
    parser.add_argument("--env_config_path", type=str, default="src/DL_analysis/config/cnn.json")
    parser.add_argument("--output_dir", type=str, default="results/nested_cv_output")
    parser.add_argument("--test_mode", action='store_true')
    
    # Path arguments (can be overriden by config, but needed for args attribute existence)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_dir_augmented", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load Environment Config
    if os.path.exists(args.env_config_path):
        with open(args.env_config_path, 'r') as f:
            full_cfg = json.load(f)
            # Support both flat config and nested 'global' config
            env_config = full_cfg.get('global', full_cfg)
            
            # Inject into args
            args.metadata_path = env_config.get('metadata_path', args.metadata_path)
            args.data_dir = env_config.get('data_dir', args.data_dir)
            args.data_dir_augmented = env_config.get('data_dir_augmented', args.data_dir_augmented)
    else:
        # Fallback or strict error
        raise FileNotFoundError(f"Environment config not found at {args.env_config_path}")
        
    nested_cv_classification(args)
