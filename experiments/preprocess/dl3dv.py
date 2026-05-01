from pathlib import Path
from tqdm import tqdm
import json
import gzip

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/DL3DV-10K")

# Loop through all scene directories

out = {}

for subfolder in root.iterdir():
    if subfolder.is_dir():  # e.g., 1K, 2K, ...
        for scene_dir in subfolder.iterdir():
            images_dir = scene_dir / "images_8"
            if images_dir.exists():
                frame_files = sorted(images_dir.glob("frame_*.png"))
                # print(f"Scene '{scene_dir.name}' has {len(frame_files)} frames:")
                out[scene_dir.name] = [{"filepath": f"{subfolder.name}/{scene_dir.name}/images_8/{f.name}", 
                                        "id": scene_dir.name+"/"+f.name} for f in frame_files]


root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl"
with open(root+"/annotations/dl3dv.json", "w") as f:
    json.dump(out, f, indent=4)  # `indent=4` makes it pretty-printed

# Save as .jgz
with gzip.open(root+"/annotations/dl3dv.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")


