
import subprocess
import time
import os
import sys

# Configuration based on user request
QUEUE = [
    # (Group1, Group2, Model)
    ('AD', 'PSP', 'alexnet'),
    ('AD', 'CBS', 'resnet'),
    ('PSP', 'CBS', 'resnet'),
    ('AD', 'CBS', 'alexnet'),
    ('PSP', 'CBS', 'alexnet'),
    ('AD', 'CBS', 'vgg16'),
    ('PSP', 'CBS', 'vgg16')
]

def wait_for_pid(pid):
    """Waits for a specific PID to terminate."""
    print(f"[SmartQueue] Waiting for existing VGG process (PID {pid}) to finish...")
    while True:
        try:
            # os.kill(pid, 0) does not kill the process, just checks if we can send a signal
            os.kill(pid, 0)
            time.sleep(60) # Check every minute
        except OSError:
            # Process dead or permission denied (usually dead if we own it)
            break
            
    print(f"\\n[SmartQueue] PID {pid} finished. Resuming queue.")

def run_task(g1, g2, model):
    print(f"\\n[SmartQueue] Starting: {g1} vs {g2} | {model}")
    output_dir = f"results/DL/{g1}_{g2}/{model}"
    
    cmd = [
        "python3", "src/DL_analysis/training/nested_cv_runner.py",
        "--group1", g1,
        "--group2", g2,
        "--model", model,
        "--output_dir", output_dir
    ]
    
    # We append to the smart_queue.log itself or a specific log
    with open("logs/orchestration/smart_queue.log", "a") as log_file:
         log_file.write(f"\\n{'='*40}\\nSTARTING: {g1}_{g2} {model}\\n{'='*40}\\n")
         subprocess.run(cmd, stdout=log_file, stderr=log_file)
         
    print(f"[SmartQueue] Finished: {g1} vs {g2} | {model}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_smart_queue.py <PID_TO_WAIT_FOR>")
        sys.exit(1)
        
    wait_pid = int(sys.argv[1])
    
    # 1. Wait
    wait_for_pid(wait_pid)
    
    # 2. Execute Queue
    for g1, g2, model in QUEUE:
        run_task(g1, g2, model)
