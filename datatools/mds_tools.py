from streaming import LocalDataset
from datasets import load_dataset, Dataset, load_from_disk
import zstandard as zstd

def read_jsonl_zst(file_path):
    with open(file_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            text = reader.read().decode('utf-8')
            return [line for line in text.splitlines() if line]

def load_source(source_type, path, secondary_path=None, hf_split="train", cache_dir=None):
    if source_type == "jsonl":
        return open(path).readlines()
    elif source_type == "jsonl.zstd":
        return read_jsonl_zst(path)
    elif source_type == "hf":
        if secondary_path is not None:
            return load_dataset(path, secondary_path, keep_in_memory=False,
                                cache_dir=cache_dir, split=hf_split)
        else:
            return load_dataset(path, keep_in_memory=False,
                                cache_dir=cache_dir, split=hf_split)
    elif source_type == "hf_local":
        return load_from_disk(path, keep_in_memory=False)
    elif source_type == "arrow":
        # Assume the arrow file is small so directly load it
        return Dataset.from_file(path)
    elif source_type == "mds":
        return LocalDataset(path)
    else:
        print(f"Unidentified file type {source_type}")
        raise NotImplementedError
