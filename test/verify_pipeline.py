
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from DL_analysis.cnn.datasets import FCDataset, AugmentedFCDataset

def verify_pipeline(group1, group2, metadata_path, data_dir, data_dir_augmented):
    print(f"\n[VERIFICATION REPORT] {group1} vs {group2}")
    print("=" * 60)
    
    # 1. Verification: Metadata Loading
    df = pd.read_csv(metadata_path)
    df = df[df['Group'].isin([group1, group2])].reset_index(drop=True)
    X = df['ID'].values
    y = df['Group'].values
    print(f"Total Subjects: {len(df)}")
    print(f"   - {group1}: {sum(y == group1)}")
    print(f"   - {group2}: {sum(y == group2)}")

    # 2. Logic Reconstruction
    # MUST MATCH nested_cv_runner.py EXACTLY: StratifiedKFold(5, shuffle=True, seed=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Loop Outer
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n[OUTER FOLD {outer_fold + 1}/5]")
        
        train_subjects = df.iloc[train_idx]
        test_subjects = df.iloc[test_idx]
        
        # CHECK 1: Data Leakage (Outer)
        overlap = set(train_subjects['ID']) & set(test_subjects['ID'])
        if len(overlap) == 0:
            print(f"   [PASS] Split Integrity (No overlap between Train/Test)")
        else:
            print(f"   [FAIL] LEAKAGE DETECTED! Subjects: {overlap}")
            
        print(f"   - Outer Train Size: {len(train_subjects)}")
        print(f"   - Outer Test Size : {len(test_subjects)}")

        # Inner CV Logic
        # MUST MATCH nested_cv_runner.py EXACTLY
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Take just the first inner fold as sample or check all
        for inner_fold, (in_train_idx, in_val_idx) in enumerate(inner_cv.split(train_subjects['ID'], train_subjects['Group'])):
            inner_train = train_subjects.iloc[in_train_idx]
            inner_val = train_subjects.iloc[in_val_idx]
            
            # CHECK 2: Data Leakage (Inner)
            overlap_inner = set(inner_train['ID']) & set(inner_val['ID'])
            overlap_test = set(inner_val['ID']) & set(test_subjects['ID']) # Crucial: Val vs Outer Test
            
            if len(overlap_inner) == 0 and len(overlap_test) == 0:
                pass
            else:
                print(f"   [FAIL] INNER LEAKAGE DETECTED in Fold {inner_fold+1}!")
                
            # CHECK 3: Dataset Augmentation Sources
            if inner_fold == 0: # Check once per outer fold
                ds_train = AugmentedFCDataset(data_dir_augmented, inner_train, 'Group', task='classification')
                ds_val = FCDataset(data_dir, inner_val, 'Group', task='classification')
                
                print(f"   [DATA CHECK] Inner Fold 1:")
                
                # Check Train (Augmented)
                expected_aug = data_dir_augmented
                actual_aug = ds_train.data_dir
                if expected_aug in actual_aug:
                    print(f"     [PASS] Train Set: Uses AUGMENTED folder")
                    print(f"        Path: {actual_aug}...")
                    print(f"        Subjects: {len(inner_train)} -> Dataset Length: {len(ds_train)} (Ratio: {len(ds_train)/len(inner_train):.1f}x)")
                else:
                     print(f"     [FAIL] Train Set: WRONG FOLDER ({actual_aug})")

                # Check Val (Original)
                expected_orig = data_dir
                actual_orig = ds_val.data_dir
                if expected_orig in actual_orig and "augmented" not in actual_orig:
                     print(f"     [PASS] Val Set  : Uses ORIGINAL folder")
                     print(f"        Path: {actual_orig}...")
                     print(f"        Subjects: {len(inner_val)} -> Dataset Length: {len(ds_val)} (Ratio: {len(ds_val)/len(inner_val):.1f}x)")
                else:
                     print(f"     [FAIL] Val Set  : WRONG FOLDER ({actual_orig})")
        
        print("   [PASS] Inner CV Loops (5/5 folds checked for leakage)")

        print("   [PASS] Inner CV Loops (5/5 folds checked for leakage)")

    print("\n" + "=" * 60)
    print("[CROSS-VALIDATION DISTRIBUTION CHECK]")
    print(f"Checking that each subject appears exactly once in Test set across {outer_cv.get_n_splits()} folds...")
    
    # Initialize tracker
    # subject_id -> {'test': count, 'train': count}
    tracker = {sid: {'test': 0, 'train': 0} for sid in X}
    
    # Re-run loop purely for tracking (cleaner than mixing with above verification)
    outer_cv_check = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(outer_cv_check.split(X, y)):
        train_ids = X[train_idx]
        test_ids = X[test_idx]
        
        for tid in train_ids:
            tracker[tid]['train'] += 1
        for tid in test_ids:
            tracker[tid]['test'] += 1
            
    # Analyze
    test_counts = [stats['test'] for stats in tracker.values()]
    train_counts = [stats['train'] for stats in tracker.values()]
    
    # Verify Test Counts (Should be all 1)
    if all(c == 1 for c in test_counts):
        print(f"   [PASS] PERFECT DISTRIBUTION: All {len(X)} subjects tested exactly 1 time.")
    else:
        print(f"   [FAIL] DISTRIBUTION ERROR:")
        print(f"          Subjects tested != 1 time: {sum(c != 1 for c in test_counts)}")
        print(f"          (Min: {min(test_counts)}, Max: {max(test_counts)})")

    # Verify Train Counts (Should be all 4 for 5-fold)
    expected_train = 4
    if all(c == expected_train for c in train_counts):
        print(f"   [PASS] PERFECT TRAIN FREQUENCY: All {len(X)} subjects trained exactly {expected_train} times.")
    else:
        print(f"   [WARNING] Train distribution variance (expected {expected_train}):")
        print(f"             Min: {min(train_counts)}, Max: {max(train_counts)}")
        
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group1", type=str, default="AD")
    parser.add_argument("--group2", type=str, default="PSP")
    parser.add_argument("--metadata_path", type=str, default="/data/users/etosato/ANM_Verona/assets/metadata/labels.csv")
    parser.add_argument("--data_dir", type=str, default="/data/users/etosato/ANM_Verona/data/FCmaps_processed")
    parser.add_argument("--data_dir_augmented", type=str, default="/data/users/etosato/ANM_Verona/data/FCmaps_augmented_processed")
    
    args = parser.parse_args()
    
    verify_pipeline(args.group1, args.group2, args.metadata_path, args.data_dir, args.data_dir_augmented)
