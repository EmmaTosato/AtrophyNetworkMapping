
import subprocess
import time
import os
import sys

# Configuration based on user request
QUEUE = [
    # 1. ResNet (Missing Groups)
    ('AD', 'CBS', 'resnet'),
    ('PSP', 'CBS', 'resnet'),
    
    # 2. AlexNet (All Groups)
    ('AD', 'PSP', 'alexnet'),
    ('AD', 'CBS', 'alexnet'),
    ('PSP', 'CBS', 'alexnet')
    
    # VGG16 for others is kept for later manually if needed
]

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
    print("Starting Smart Queue immediately (No PID wait)...")
    
    # Execute Queue
    for g1, g2, model in QUEUE:
        run_task(g1, g2, model)
