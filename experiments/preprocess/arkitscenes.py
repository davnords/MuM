import os
import re
from pathlib import Path
from tqdm import tqdm
import json
import gzip

def natural_sort_key(s):
    """Split string into numbers and non-numbers for natural sorting"""
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/ARKitScenes_data/3dod/Training/")

# Output structure
out = {}
chunk_size = 75

# Loop through all sequence directories
for sequence_name in os.listdir(root):
    sequence_path = root / sequence_name / f"{sequence_name}_frames" / "lowres_wide"
    
    if sequence_path.exists():
        print(f"Processing sequence: {sequence_name}")
        
        # Get all files and sort them naturally
        files = [f for f in os.listdir(sequence_path) if os.path.isfile(sequence_path / f)]
        files.sort(key=natural_sort_key)
        
        num_frames = len(files)
        
        if num_frames == 0:
            print(f"  No files found in {sequence_name}, skipping...")
            continue
        
        # Since the images are taken in a sequence we will just chunk up the sequences
        sequences = []
        
        if num_frames <= chunk_size:
            # If we have fewer frames than chunk_size, create one sequence
            sequences.append(files)
        else:
            # Calculate how many full chunks we can take, stopping before the last chunk
            num_full_chunks = (num_frames - 1) // chunk_size # leave room for overflow in last chunk
            
            for i in range(num_full_chunks - 1):
                sequences.append(files[i * chunk_size: (i + 1) * chunk_size])
            
            # Last chunk gets the rest of the frames
            sequences.append(files[(num_full_chunks - 1) * chunk_size:])
        
        # Create entries for each sequence
        for i, seq in enumerate(sequences):
            sequence_key = f"{sequence_name}_{i}"
            out[sequence_key] = [
                {
                    "filepath": f"{sequence_name}/{sequence_name}_frames/lowres_wide/{f}",
                    "id": f"{sequence_name}/{f}"
                } 
                for f in seq
            ]
        
        print(f"  Created {len(sequences)} sequences for {sequence_name} with {num_frames} total frames")
    else:
        print(f"  Sequence path does not exist: {sequence_path}")

# Save the output
output_root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl"
os.makedirs(f"{output_root}/annotations", exist_ok=True)

# Save as JSON
with open(f"{output_root}/annotations/ARKitScenes.json", "w") as f:
    json.dump(out, f, indent=4)

# Save as compressed JSON
with gzip.open(f"{output_root}/annotations/ARKitScenes.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

total_images = sum(len(v) for v in out.values())
print(f"\nProcessed {len(out)} sequences with a total of {total_images} images.")
print(f"Output saved to {output_root}/annotations/ARKitScenes.json and ARKitScenes.jgz")