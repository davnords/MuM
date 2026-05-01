from pathlib import Path
from tqdm import tqdm
import json
import gzip

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/blendedmvs")

# Loop through all scene directories

out = {}

for scene_dir in root.iterdir():
    images_dir = scene_dir / "blended_images"
    if images_dir.exists():
        frame_files = sorted(f for f in images_dir.glob("*.jpg") if "_masked" not in f.name)
        out[scene_dir.name] = [{"filepath": f"{scene_dir.name}/blended_images/{f.name}", 
                                "id": scene_dir.name+"/"+f.name} for f in frame_files]


root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl"
with open(root+"/annotations/blendedmvs.json", "w") as f:
    json.dump(out, f, indent=4)  # `indent=4` makes it pretty-printed

# Save as .jgz
with gzip.open(root+"/annotations/blendedmvs.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")


