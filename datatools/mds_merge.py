import json
import sys
import os
from streaming import LocalDataset

index_name = "index.json"

def merge_index(root_dir): 
    if os.path.exists(os.path.join(root_dir, index_name)):
        # print(os.path.join(root_dir, index_name) + " already exists. Return")
        return
    
    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for d in subfolders:
        merge_index(os.path.join(root_dir, d))

    print(f"Merge the index for {root_dir}")
    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and os.path.exists(os.path.join(root_dir, d, index_name))]
    print(f"| Find the following subfolders with index.json: {subfolders}")

    if len(subfolders) > 0:
        shards = []
        for d in subfolders:
            subindex = json.load(open(os.path.join(root_dir, d, index_name)))
            for shard in subindex["shards"]:
                shard['raw_data']['basename'] = os.path.join(d, shard['raw_data']['basename'])
                shards.append(shard)
        new_index = {
            'version': 2,
            'shards': shards,
        }
        json.dump(new_index, open(os.path.join(root_dir, index_name), 'w'), indent=4)
        print(f"| Successfully wrote the new index.json to {os.path.join(root_dir, index_name)}!")
        print(f"| Test: loading LocalDataset from {root_dir}")
        ds = LocalDataset(root_dir)
        print(f"| Successfully loaded {len(ds)} data")
        print(f"| Try reading a random data {ds[len(ds) // 2]}")

if __name__ == "__main__":
    root_dir = sys.argv[1]
    merge_index(root_dir)
