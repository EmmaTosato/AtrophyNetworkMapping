
import argparse
import subprocess
import os
import sys

# Standard configurations
ALL_PAIRS = [
    ('AD', 'PSP'),
    ('AD', 'CBS'),
    ('PSP', 'CBS')
]

ALL_MODELS = ['resnet', 'vgg16', 'alexnet']

def run_command(cmd, log_file=None):
    """Run a shell command and stream output."""
    print(f"Executing: {' '.join(cmd)}")
    
    # We want to stream output to console so user sees progress
    # But usually nested_cv_runner.py redirects its OWN output to run.log
    # So we just wait for it to finish.
    
    process = subprocess.Popen(
        cmd, 
        stdout=sys.stdout,
        stderr=sys.stderr,
        universal_newlines=True
    )
    
    return_code = process.wait()
    if return_code != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        sys.exit(return_code)

def main():
    parser = argparse.ArgumentParser(description="Automated Benchmark Runner for Nested CV")
    
    parser.add_argument("--pairs", nargs='+', default=['all'], 
                        help="Pairs to run (format: AD_PSP). Use 'all' for [AD_PSP, AD_CBS, PSP_CBS]")
    
    parser.add_argument("--models", nargs='+', default=['all'],
                        help="Models to run. Use 'all' for [resnet, vgg16, alexnet]")
    
    parser.add_argument("--dry_run", action='store_true', 
                        help="Print commands without executing them")
    
    parser.add_argument("--test_mode", action='store_true',
                        help="Pass --test_mode to the runner for quick verification")
    
    args = parser.parse_args()
    
    # 1. Parse Pairs
    if 'all' in args.pairs:
        pairs_to_run = ALL_PAIRS
    else:
        pairs_to_run = []
        for p in args.pairs:
            # Expect format GROUP1_GROUP2
            if '_' not in p:
                print(f"Invalid pair format: {p}. Expected G1_G2 (e.g. AD_PSP)")
                sys.exit(1)
            g1, g2 = p.split('_')
            pairs_to_run.append((g1, g2))
            
    # 2. Parse Models
    if 'all' in args.models:
        models_to_run = ALL_MODELS
    else:
        models_to_run = args.models
        
    # 3. Base Dir
    # We assume we are running from project root
    # 3. Base Dir
    # We assume we are running from project root
    base_output_dir = "results/DL"
    
    print("="*60)
    print(f"BENCHMARK RUNNER CONFIGURATION")
    print(f"Pairs : {[f'{g1} vs {g2}' for g1,g2 in pairs_to_run]}")
    print(f"Models: {models_to_run}")
    print(f"Test Mode: {args.test_mode}")
    print("="*60)
    
    # 4. Loop and Execute
    for g1, g2 in pairs_to_run:
        for model in models_to_run:
            
            pair_name = f"{g1}_{g2}"
            output_dir = os.path.join(base_output_dir, pair_name, model)
            
            print(f"\n---> Starting Run: {pair_name} | {model}")
            print(f"     Output Dir: {output_dir}")
            
            cmd = [
                "python3", "src/DL_analysis/training/nested_cv_runner.py",
                "--group1", g1,
                "--group2", g2,
                "--model", model,
                "--output_dir", output_dir
            ]
            
            if args.test_mode:
                cmd.append("--test_mode")
            
            if args.dry_run:
                print(f"DRY RUN CMD: {' '.join(cmd)}")
            else:
                run_command(cmd)
                print(f"---> Finished Run: {pair_name} | {model}")

    print("\nAll selected benchmarks completed.")

if __name__ == "__main__":
    main()
