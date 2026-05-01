from pathlib import Path
from tqdm import tqdm
import json
import gzip

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/scannetpp_data/data")

# Loop through all scene directories

out = {}

chunk_size = 100


for scene_dir in root.iterdir():
    dslr_dir = scene_dir / "dslr"
    if dslr_dir.exists():
        train_test_list = dslr_dir / "train_test_lists.json"
        if train_test_list.exists():
            with open(train_test_list, 'r') as f:
                train_test_data = json.load(f)

            frames = train_test_data['train']
            # Maybe resized undistorted images are too high resolution?
            num_frames = len(frames)

            # Since the images are taken in a sequence we will just chunk up the sequences

            sequences = []
            # Calculate how many full chunks we can take, stopping before the last chunk
            num_full_chunks = (num_frames - 1) // chunk_size  # leave room for overflow in last chunk

            for i in range(num_full_chunks - 1):
                sequences.append(frames[i * chunk_size: (i + 1) * chunk_size])

            # Last chunk gets the rest of the frames
            sequences.append(frames[(num_full_chunks - 1) * chunk_size:])

            for i, seq in enumerate(sequences):
                out[scene_dir.name+"_"+str(i)] = [{"filepath": f"{scene_dir.name}/dslr/resized_undistorted_images/{f}", 
                                        "id": f"{scene_dir.name}/{f}"} for f in seq]

            print(f"  Created {len(sequences)} sequences for {scene_dir.name}")

root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl"
with open(root+"/annotations/scannetpp.json", "w") as f:
    json.dump(out, f, indent=4)  # `indent=4` makes it pretty-printed

with gzip.open(root+"/annotations/scannetpp.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")


