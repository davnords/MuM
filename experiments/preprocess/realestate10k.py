from pathlib import Path
from tqdm import tqdm
import json
import gzip

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/RealEstate10K_Downloader/dataset/train")

# Loop through all scene directories

out = {}

for scene_dir in root.iterdir():
    frame_files = sorted(scene_dir.glob("*.png"))
    out[scene_dir.name] = [{"filepath": f"{scene_dir.name}/{f.name}", 
                            "id": scene_dir.name+"/"+f.name} for f in frame_files]


root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl"

# Save as .jgz
with gzip.open(root+"/annotations/realestate10k.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")


