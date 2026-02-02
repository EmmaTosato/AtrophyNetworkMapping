#!/usr/bin/env python3
"""
Batch script to run all regression pipeline combinations.
16 combinations: 2 datasets × 2 targets × 2 covariate settings × 2 group settings
"""
import os
import sys
import itertools
from datetime import datetime

# Add src to path
# Script is in src/ML_analysis/analysis/
# We need to add 'src' folder to path, which is 2 levels up from here
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from ML_analysis.analysis.regression import main_regression
from ML_analysis.loading.config import ConfigLoader

import argparse

# Load default config paths
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
default_run_config = os.path.join(project_root, "src/ML_analysis/config/ml_run_all_config.json")
default_ml_config = os.path.join(project_root, "src/ML_analysis/config/ml_config.json")

# Argparse
parser = argparse.ArgumentParser(description="Run all regressions batch script")
parser.add_argument("--run_config", default=default_run_config, help="Path to ml_run_all_config.json")
parser.add_argument("--config", default=default_ml_config, help="Path to ml_config.json")
args_cli = parser.parse_args() # Rename to avoid conflict with 'args' dict in loop

run_config_path = args_cli.run_config

DATASETS = ['voxel', 'networks']
TARGETS = ['CDR_SB', 'MMSE']
FLAG_COVARIATES = [False, True]
GROUP_REGRESSION = [False, True]

if os.path.exists(run_config_path):
    try:
        import json
        with open(run_config_path, "r") as f:
            run_cfg = json.load(f)
        grid = run_cfg.get("regression", {}).get("grid", {})
        DATASETS = grid.get("datasets", DATASETS)
        TARGETS = grid.get("targets", TARGETS)
        FLAG_COVARIATES = grid.get("flag_covariates", FLAG_COVARIATES)
        GROUP_REGRESSION = grid.get("group_regression", GROUP_REGRESSION)
        print(f"Loaded regression grid from: {run_config_path}")
    except Exception as e:
        print(f"Error loading run config: {e}. Using defaults.")
else:
    print("Run config not found. Using defaults.")

# Data paths now loaded via ConfigLoader
# DATA_PATHS and META_PATH constants removed to avoid hardcoding

def load_data(dataset_type):
    """Load data based on dataset type using ConfigLoader to resolve paths."""
    loader = ConfigLoader()
    
    # We need to manually resolve the path for the requested dataset_type
    # because ConfigLoader init loads whatever is in ml_config.json, which might be different.
    # Fortunately, ConfigLoader exposes _resolve_data_path method (helper).
    
    # Get all paths from loaded config
    paths = loader.args
    
    # Manually resolve for the specific requested type (defaulting threshold to False/None for regression base)
    if dataset_type == 'voxel':
        # Default regression uses 'df_masked' (gm_mask) typically
        df_path = paths.get('df_masked')
        df_input = pd.read_pickle(df_path)
    elif dataset_type == 'networks':
        # Default regression uses 'yeo_noThr'
        df_path = paths.get('yeo_noThr')
        df_input = pd.read_csv(df_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
        
    meta_path = paths.get('df_meta')
    df_meta = pd.read_csv(meta_path)
    
    # Merge metadata
    if 'Group' not in df_input.columns:
        df_input = pd.merge(df_input, df_meta, on='ID', how='inner')
    
    return df_input, df_meta


def run_combination(dataset_type, target, flag_cov, group_reg):
    """Run a single combination."""
    
    # Build args
    args = {
        'dataset_type': dataset_type,
        'threshold': False,
        'task_type': 'regression',
        'umap': dataset_type == 'voxel',  # UMAP only for voxel
        'target_variable': target,
        'flag_covariates': flag_cov,
        'group_regression': group_reg,
        'group_col': 'Group',
        'covariates': ['Age', 'Sex', 'Education'] if flag_cov else None,
        'y_scaled': True, 
        'inverse_transform_y': True, # Default to True (can be overridden by config loading)
        'color_by_group': True,
        'group_name': 'Group',
        'plot_regression': True,
        'save_flag': True,
        'save_flag': True,
        # Load default output_dir from ml_config.json or fallback
        'output_dir': 'results/ML/', 
    }
    
    # Try to load output_dir from actual config file if possible, 
    # but since this script builds args manually, we keep it simple or read the file.
    # Ideally:
    # with open("src/ML_analysis/config/ml_config.json") as f:
    #    cfg = json.load(f)
    #    args['output_dir'] = cfg.get('fixed_parameters', {}).get('output_dir', args['output_dir'])
    
    # For now, let's keep it robust as requested:
    try:
        import json
        with open(args_cli.config) as f:
             cfg = json.load(f)
             fixed = cfg.get('fixed_parameters', {})
             args['output_dir'] = fixed.get('output_dir', 'results/ML/')
             # CRITICAL FIX: Load standardization flags from config
             args['y_scaled'] = fixed.get('y_scaled', True)
             args['inverse_transform_y'] = fixed.get('inverse_transform_y', True)
    except Exception as e:
        print(f"WARNING: Failed to load config in run_combination. Using defaults. Error: {e}")
        pass
    
    # Status
    cov_str = 'cov' if flag_cov else 'no_cov'
    group_str = 'group' if group_reg else 'global'
    combo_name = f"{dataset_type}_{target}_{cov_str}_{group_str}"
    
    print("\n" + "=" * 70)
    print(f"RUNNING: {combo_name}")
    print("=" * 70)
    print(f"  Dataset: {dataset_type} | UMAP: {args['umap']}")
    print(f"  Target: {target}")
    print(f"  Covariates: {args['covariates']}")
    print(f"  Group Regression: {group_reg}")
    print("-" * 70)
    
    try:
        df_input, df_meta = load_data(dataset_type)
        main_regression(args, df_input, df_meta)
        print(f"✅ COMPLETED: {combo_name}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {combo_name}")
        print(f"   Error: {str(e)}")
        return False


def main():
    start_time = datetime.now()
    
    print("=" * 70)
    print("BATCH REGRESSION PIPELINE")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Generate all combinations
    combinations = list(itertools.product(DATASETS, TARGETS, FLAG_COVARIATES, GROUP_REGRESSION))
    total = len(combinations)
    
    print(f"Total combinations to run: {total}")
    print("-" * 70)
    
    results = []
    for i, (dataset, target, flag_cov, group_reg) in enumerate(combinations, 1):
        print(f"\n[{i}/{total}]", end="")
        success = run_combination(dataset, target, flag_cov, group_reg)
        results.append({
            'dataset': dataset,
            'target': target,
            'covariates': flag_cov,
            'group_regression': group_reg,
            'success': success
        })
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("BATCH COMPLETED")
    print("=" * 70)
    print(f"Duration: {duration}")
    print(f"Successful: {sum(r['success'] for r in results)}/{total}")
    
    # Show failures
    failures = [r for r in results if not r['success']]
    if failures:
        print("\nFailed combinations:")
        for f in failures:
            print(f"  - {f['dataset']}_{f['target']}_{'cov' if f['covariates'] else 'no_cov'}_{'group' if f['group_regression'] else 'global'}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
