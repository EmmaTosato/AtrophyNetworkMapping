import argparse
import subprocess
import os
import sys
import json

def run_command(cmd, log_file=None):
    """Run a shell command and stream output."""
    print(f"Executing: {' '.join(cmd)}")
    
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
    # Dynamically resolve project root (3 levels up from this script: src/DL_analysis/training -> src/DL/ -> src/ -> root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    
    # Default paths handled dynamically
    default_config = os.path.join(project_root, "src/DL_analysis/config/cnn.json")
    default_grid = os.path.join(project_root, "src/DL_analysis/config/cnn_grid.json")
    
    parser = argparse.ArgumentParser(description="Automated Benchmark Runner (Config Driven)")
    parser.add_argument("--config", type=str, default=default_config,
                        help="Path to runner configuration JSON (Experiments + Environment)")
    parser.add_argument("--grid_config", type=str, default=default_grid,
                        help="Path to hyperparameter grid JSON")
    parser.add_argument("--dry_run", action='store_true', help="Print commands only")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        job_config = json.load(f)
        
    base_output_dir = job_config.get('global', {}).get('base_output_dir', 'results/DL')
    
    # If base_output_dir is relative, make it relative to project root for consistency
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(project_root, base_output_dir)
        
    global_test_mode = job_config.get('global', {}).get('test_mode', False)
    
    experiments = job_config.get('experiments', [])
    
    print("="*60)
    print(f"BENCHMARK RUNNER (JSON Config)")
    print(f"Project Root       : {project_root}")
    print(f"Experiments Config : {args.config}")
    print(f"Hyperparams Config : {args.grid_config}")
    print(f"Base Output        : {base_output_dir}")
    print(f"Test Mode          : {global_test_mode}")
    print(f"Jobs found         : {len(experiments)}")
    print("="*60)
    
    # Path to the child script (run_nested_cv.py is in the same dir as this script)
    nested_cv_script = os.path.join(script_dir, "run_nested_cv.py")
    
    for i, exp in enumerate(experiments):
        g1 = exp['group1']
        g2 = exp['group2']
        models = exp['models']
        
        print(f"\n[Experiment {i+1}] {g1} vs {g2} -> Models: {models}")
        
        for model in models:
            pair_name = f"{g1}_{g2}"
            output_dir = os.path.join(base_output_dir, pair_name, model)
            
            cmd = [
                "python3", nested_cv_script,
                "--group1", g1,
                "--group2", g2,
                "--model", model,
                "--output_dir", output_dir,
                "--env_config_path", args.config,
                "--config_path", args.grid_config
            ]
            
            if global_test_mode:
                cmd.append("--test_mode")
                
            if args.dry_run:
                print(f"  [DRY RUN] {' '.join(cmd)}")
            else:
                run_command(cmd)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
