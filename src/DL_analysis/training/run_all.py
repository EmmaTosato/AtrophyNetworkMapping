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
    # Dynamically resolve project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    
    # Default Paths
    default_batch_config = os.path.join(project_root, "src/DL_analysis/config/run_all.json")
    
    parser = argparse.ArgumentParser(description="Generic Batch Runner (Config Driven)")
    parser.add_argument("--config", type=str, default=default_batch_config,
                        help="Path to the batch execution JSON")
    parser.add_argument("--dry_run", action='store_true', help="Print commands only")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Batch config file not found: {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        batch_config = json.load(f)
        
    script_rel_path = batch_config.get('script_path')
    config_rel_path = batch_config.get('config_path')
    experiments = batch_config.get('experiments', [])
    
    # Resolve Paths (Assume relative to Project Root)
    # Note: run_all.json defines paths relative to root usually.
    if not os.path.isabs(script_rel_path):
        script_full_path = os.path.join(project_root, script_rel_path)
    else:
        script_full_path = script_rel_path
        
    if not os.path.isabs(config_rel_path):
        config_full_path = os.path.join(project_root, config_rel_path)
    else:
        config_full_path = config_rel_path
        
    if not os.path.exists(script_full_path):
        print(f"Target script not found: {script_full_path}")
        sys.exit(1)

    print("="*60)
    print(f"BATCH RUNNER (Generic)")
    print(f"Batch Config   : {args.config}")
    print(f"Target Script  : {script_full_path}")
    print(f"Base State     : {config_full_path}")
    print(f"Job Count      : {len(experiments)}")
    print("="*60)
    
    for i, exp in enumerate(experiments):
        # Extract params from the experiment block
        g1 = exp.get('group1')
        g2 = exp.get('group2')
        tuning = exp.get('tuning') # True/False/None
        
        # Support 'models' (list) or 'model' (string)
        models = exp.get('models')
        if models is None:
            models = [exp.get('model')]
            
        print(f"\n[Job {i+1}] {g1} vs {g2} -> Models: {models}")
        
        for model in models:
            # Construct Command
            cmd = [
                "python3", script_full_path,
                "--config_path", config_full_path,
                "--group1", g1,
                "--group2", g2,
                "--model", model
            ]
            
            # Pass tuning flag only if True
            # (If False, we omit it, relying on script default/absence logic)
            # Actually, per our new logic in nested_cv:
            # "if args.tuning: experiment['tuning'] = True"
            # So passing --tuning activates it. NOT passing it leaves it as per config default (False).
            if tuning is True:
                cmd.append("--tuning")
            
            if args.dry_run:
                print(f"  [DRY RUN] {' '.join(cmd)}")
            else:
                run_command(cmd)

    print("\nAll batch jobs completed.")

if __name__ == "__main__":
    main()
