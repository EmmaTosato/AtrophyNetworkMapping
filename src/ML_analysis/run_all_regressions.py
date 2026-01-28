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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from ML_analysis.analysis.regression import main_regression

# Configuration combinations
DATASETS = ['voxel', 'networks']
TARGETS = ['CDR_SB', 'MMSE']
FLAG_COVARIATES = [False, True]
GROUP_REGRESSION = [False, True]

# Data paths
DATA_PATHS = {
    'voxel': 'data/dataframes/fdc/df_gm.pkl',
    'networks': 'data/dataframes/networks/networks_noTHR.csv'
}
META_PATH = 'assets/metadata/df_meta.csv'


def load_data(dataset_type):
    """Load data based on dataset type."""
    if dataset_type == 'voxel':
        df_input = pd.read_pickle(DATA_PATHS['voxel'])
    else:
        df_input = pd.read_csv(DATA_PATHS['networks'])
    
    df_meta = pd.read_csv(META_PATH)
    
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
        'y_log_transform': False,
        'color_by_group': True,
        'group_name': 'Group',
        'plot_regression': True,
        'save_flag': True,
        'output_dir': 'results/ML/',
    }
    
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
