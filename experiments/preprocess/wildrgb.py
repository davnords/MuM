import os
import re
from pathlib import Path
from tqdm import tqdm
import json
import gzip

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/wildrgbd/")

# Output structure
out = {}

# Loop through all sequence directories
for category_name in tqdm(os.listdir(root)):
    category_path = root / category_name / "scenes"

    print(f"Processing category: {category_name}")

    for sequence_name in os.listdir(category_path):
        try:
            sequence_path = category_path / sequence_name / "rgb"

            
            files = [f for f in os.listdir(sequence_path)]
            
            num_frames = len(files)
            
            if num_frames == 0:
                print(f"  No files found in {sequence_name}, skipping...")
                continue
            
            # Since the images are taken in a sequence we will just chunk up the sequences

            if len(files)>23:
                sequence_key = f"{category_name}_{sequence_name}_"
                out[sequence_key] = [
                    {
                        "filepath": f"{category_name}/scenes/{sequence_name}/rgb/{f}",
                        "id": f"{category_name}/{sequence_name}/{f}"
                    } 
                    for f in files
                ]

                print(f"  Created a sequence for category {category_name} with {num_frames} total frames")
            else:
                print(f"  Skipped {sequence_name} in category {category_name} (not enough frames)")
        except FileNotFoundError:
            print(f"  Directory not found for {sequence_name} in category {category_name}, skipping...")
            continue

# Save the output
output_root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl"
os.makedirs(f"{output_root}/annotations", exist_ok=True)

# Save as JSON
with open(f"{output_root}/annotations/wildrgb.json", "w") as f:
    json.dump(out, f, indent=4)

# Save as compressed JSON
with gzip.open(f"{output_root}/annotations/wildrgb.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

total_images = sum(len(v) for v in out.values())
print(f"\nProcessed {len(out)} sequences with a total of {total_images} images.")
print(f"Output saved to {output_root}/annotations/wildrgb.json and wildrgb.jgz")