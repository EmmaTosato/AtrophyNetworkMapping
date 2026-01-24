import pandas as pd
import sys

def csv_to_md(csv_path, output_path):
    df = pd.read_csv(csv_path)
    # Fill NaN with empty string
    df.fillna("", inplace=True)
    
    with open(output_path, "w") as f:
        f.write("# Permutation Test Details (100 Iterations)\n\n")
        f.write("Detailed statistics for the permutation tests. "
                "Shows the comparison between the model's true accuracy vs. the accuracy obtained on shuffled labels (Null Distribution).\n\n")
        f.write(df.to_markdown(index=False))
        
    print(f"Generated {output_path}")

if __name__ == "__main__":
    csv_to_md("results/ML/summary_permutation.csv", "doc/ML/permutation_details_100.md")
