import pandas as pd
import numpy as np
from ML_analysis.loading.config import ConfigLoader
import sys
import os

# Add src to path to allow imports
sys.path.append("src")

try:
    print("Loading configuration...")
    loader = ConfigLoader()
    
    # Load raw inputs individually to check them
    print("\n--- Inspecting Dataframes ---")
    
    # Replicate loading logic from ConfigLoader.load_all() manually to inspect before merge
    dataset_type = loader.args["dataset_type"]
    df_path = loader.args["df_path"]
    meta_path = loader.args["df_meta"]
    
    print(f"Dataset Type: {dataset_type}")
    print(f"Data Path: {df_path}")
    print(f"Meta Path: {meta_path}")

    # Load Feature Data
    if dataset_type == "voxel":
        df_features = pd.read_pickle(df_path)
    else:
        df_features = pd.read_csv(df_path)
        
    print(f"\nFeature DF shape: {df_features.shape}")
    nan_features = df_features.isna().sum().sum()
    print(f"NaNs in Feature DF: {nan_features}")
    if nan_features > 0:
        print("Columns with NaNs in Features:")
        print(df_features.columns[df_features.isna().any()].tolist())

    # Load Metadata
    df_meta = pd.read_csv(meta_path)
    print(f"\nMetadata DF shape: {df_meta.shape}")
    nan_meta = df_meta.isna().sum().sum()
    print(f"NaNs in Metadata DF: {nan_meta}")
    if nan_meta > 0:
        print("Columns with NaNs in Metadata:")
        print(df_meta.columns[df_meta.isna().any()].tolist())
        # Show rows with NaNs
        print("\nRows with NaNs in Metadata:")
        print(df_meta[df_meta.isna().any(axis=1)])

    # Simulate Merge
    print("\n--- Simulating Merge ---")
    df_merged = pd.merge(df_features, df_meta, on="ID", how="inner")
    print(f"Merged DF shape: {df_merged.shape}")
    nan_merged = df_merged.isna().sum().sum()
    print(f"NaNs in Merged DF: {nan_merged}")
    
    if nan_merged > 0:
        print("Columns with NaNs in Merged DF:")
        print(df_merged.columns[df_merged.isna().any()].tolist())

except Exception as e:
    print(f"Error: {e}")
