#!/usr/bin/env python3
"""
Test script for regression pipeline with group-specific colors.
Tests: networks, umap, CDR_SB, no covariates, group_regression=true
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from ML_analysis.analysis.regression import main_regression

def main():
    # Build args manually to avoid pickle issues with voxel data
    args = {
        'dataset_type': 'networks',
        'threshold': False,
        'task_type': 'regression',
        'umap': False,  # Networks data is already low-dimensional
        'target_variable': 'CDR_SB',
        'flag_covariates': False,
        'group_regression': True,
        'group_col': 'Group',
        'output_dir': 'tests/results/regression_color_test/',
        'covariates': None,
        'y_log_transform': False,
        'color_by_group': True,
        'group_name': 'Group',
        'plot_regression': True,
        'save_flag': True,
    }
    
    # Load data directly
    df_input = pd.read_csv('data/dataframes/networks/networks_noTHR.csv')
    df_meta = pd.read_csv('assets/metadata/df_meta.csv')
    
    # Merge metadata into df_input
    df_input = pd.merge(df_input, df_meta, on='ID', how='inner')
    
    print("=" * 60)
    print("TEST: Regression with Group-Specific Colors")
    print("=" * 60)
    print(f"Dataset: {args['dataset_type']}")
    print(f"UMAP: {args['umap']}")
    print(f"Target: {args['target_variable']}")
    print(f"Covariates: {args['flag_covariates']}")
    print(f"Group Regression: {args['group_regression']}")
    print(f"Output: {args['output_dir']}")
    print(f"Subjects: {len(df_input)}")
    print("=" * 60)
    
    # Create output dir
    os.makedirs(args['output_dir'], exist_ok=True)
    
    # Run regression
    main_regression(args, df_input, df_meta)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED - Check results in tests/results/")
    print("=" * 60)

if __name__ == "__main__":
    main()
