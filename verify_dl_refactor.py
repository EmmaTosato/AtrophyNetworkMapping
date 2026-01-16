
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    print("Testing DL imports...")
    from DL_analysis.cnn.datasets import FCDataset
    from DL_analysis.cnn.models import ResNet3D
    from DL_analysis.training.train import train
    from DL_analysis.testing.test import evaluate
    from DL_analysis.utils.cnn_utils import resolve_split_csv_path
    print("DL imports successful.")

    print("Testing Config Path...")
    config_path = "src/DL_analysis/config/cnn_config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Config loaded. Model type: {cfg.get('model_type', 'Unknown')}")
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    print("DL Verification successful!")

except Exception as e:
    print(f"DL Verification FAILED: {e}")
    sys.exit(1)
