
import os
import shutil
import sys

def main():
    root_dir = "."
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
        
    print(f"Cleaning __pycache__ in: {os.path.abspath(root_dir)}")
    
    count = 0
    for root, dirs, files in os.walk(root_dir):
        if "__pycache__" in dirs:
            path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(path)
                print(f"Removed: {path}")
                count += 1
            except Exception as e:
                print(f"Error removing {path}: {e}")
                
    print(f"Finished. Removed {count} __pycache__ directories.")

if __name__ == "__main__":
    main()
