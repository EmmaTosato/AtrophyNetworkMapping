#!/usr/bin/env python3
"""
Script to re-save pickle files with current numpy version.
This fixes compatibility issues when pickle was saved with a different numpy version.
"""
import os
import sys
import pickle
import pandas as pd

# Workaround for numpy version mismatch
import numpy as np

# Monkey-patch to handle numpy._core -> numpy.core
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect numpy._core to numpy.core for older numpy versions
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_pickle_compat(path):
    """Load pickle with numpy version compatibility."""
    with open(path, 'rb') as f:
        try:
            return pd.read_pickle(path)
        except ModuleNotFoundError:
            f.seek(0)
            return NumpyUnpickler(f).load()


def resave_pickle(input_path, output_path=None):
    """Re-save pickle with current numpy version."""
    if output_path is None:
        # Create backup and overwrite
        backup_path = input_path + '.bak'
        output_path = input_path
    else:
        backup_path = None
    
    print(f"Loading: {input_path}")
    df = load_pickle_compat(input_path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns[:5].tolist()}...")
    
    if backup_path:
        print(f"Creating backup: {backup_path}")
        os.rename(input_path, backup_path)
    
    print(f"Saving: {output_path}")
    df.to_pickle(output_path)
    print("Done!")
    
    return df


if __name__ == "__main__":
    # List of pickle files to convert
    pickle_files = [
        'data/dataframes/fdc/df_gm.pkl',
        'data/dataframes/fdc/df_har.pkl',
        'data/dataframes/fdc/df_thr01_gm.pkl',
        'data/dataframes/fdc/df_thr01_har.pkl',
        'data/dataframes/fdc/df_thr02_gm.pkl',
        'data/dataframes/fdc/df_thr02_har.pkl',
    ]
    
    print("=" * 60)
    print("Pickle Re-Saver (numpy compatibility fix)")
    print(f"Current numpy version: {np.__version__}")
    print("=" * 60)
    
    for pkl_path in pickle_files:
        if os.path.exists(pkl_path):
            print(f"\nProcessing: {pkl_path}")
            try:
                resave_pickle(pkl_path)
                print("✅ Success")
            except Exception as e:
                print(f"❌ Failed: {e}")
        else:
            print(f"\n⚠️ Not found: {pkl_path}")
    
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
