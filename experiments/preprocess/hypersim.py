from pathlib import Path
from tqdm import tqdm
import json
import gzip

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/ml-hypersim/contrib/99991/downloads")

# Loop through all scene directories

out = {}

for scene_dir in root.iterdir():
    img_dir = scene_dir / "images/scene_cam_00_final_preview"
    if img_dir.exists():
        frame_files = sorted(img_dir.glob("*.jpg"))
        out[scene_dir.name] = [{"filepath": f"{scene_dir.name}/images/scene_cam_00_final_preview/{f.name}", 
                                "id": scene_dir.name+"/"+f.name} for f in frame_files]


root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl"

# Save as .jgz
with gzip.open(root+"/annotations/hypersim.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

# print(out)
print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")


