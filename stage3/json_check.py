import glob
import json
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_sequences():
    # --- UPDATED PATH FOR YOUR STRUCTURE ---
    # This pattern accounts for the nested folders: stage3/input/stage2_output/Fall/Bed/...
    path_pattern = os.path.join("stage3", "input", "stage2_output", "**", "*.json")
    
    # Absolute path for debugging print
    search_root = os.path.abspath(os.path.join("stage3", "input", "stage2_output"))
    print(f"Searching in: {search_root}")
    
    lengths = []
    # recursive=True is critical to find files in subfolders like 'Fall/Bed/'
    files = glob.glob(path_pattern, recursive=True)
    
    print(f"Found {len(files)} JSON files.")

    if not files:
        print("\n[!] No JSONs found. Please ensure you are running the script from the 'D:\\fall_detection' folder.")
        return

    for p in files:
        try:
            with open(p, 'r') as f:
                doc = json.load(f)
                # Count the number of frames present in the 'frame_data' list
                frame_count = len(doc.get("frame_data", []))
                lengths.append(frame_count)
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue

    lengths = np.array(lengths)
    
    # Statistical analysis
    min_len = np.min(lengths)
    max_len = np.max(lengths)
    mean_len = np.mean(lengths)
    p95 = np.percentile(lengths, 95) # 95% of clips are shorter than this

    print(f"\n--- Sequence Length Stats ---")
    print(f"Total Files Processed: {len(lengths)}")
    print(f"Min Frames in a clip:  {min_len}")
    print(f"Max Frames in a clip:  {max_len}")
    print(f"Mean Frames per clip:  {mean_len:.2f}")
    print(f"95th Percentile:       {p95:.2f}")
    print(f"-----------------------------\n")
    
    suggested_seq_len = int(np.ceil(p95))
    print(f"SUGGESTED SEQ_LEN: {suggested_seq_len}") 
    
    # Visualize the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=range(min_len, max_len + 2), color='skyblue', edgecolor='black', align='left')
    plt.axvline(p95, color='red', linestyle='dashed', linewidth=2, label=f'95th Percentile ({p95:.1f})')
    plt.title('Distribution of Frame Counts (Key Frames Only)')
    plt.xlabel('Number of Frames')
    plt.ylabel('Frequency (Number of Clips)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    analyze_sequences()