
import os
import re
import sys

def remove_emojis_from_text(text):
    # Regex for a broad range of emojis
    # This covers many common ranges including specific symbol ranges often used as emojis
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"  # Dingbats
        "\U000024c2-\U0001f251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = remove_emojis_from_text(content)
        
        if content != new_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Cleaned: {filepath}")
        else:
            # print(f"Skipped (no emojis): {filepath}")
            pass
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

def main():
    root_dir = "."
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    
    print(f"Scanning for emojis in: {os.path.abspath(root_dir)}")
    
    extensions = {'.py', '.md', '.txt', '.json'}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and .git
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                process_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
