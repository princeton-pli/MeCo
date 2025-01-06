import argparse
from streaming import MDSWriter, LocalDataset
from datasets import load_dataset, Dataset, load_from_disk
import os
from llama3_tokenizer import Tokenizer
import json
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import sys
from mds_tools import load_source
import random
import re
import tldextract

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--job_id", type=int, default=0, help="Should pass $SLURM_ARRAY_TASK_ID if using SLURM array job")
parser.add_argument("--slurm_array_as_job_id", action="store_true", help="Use $SLURM_ARRAY_TASK_ID to override the job id")
parser.add_argument("--job_start_idx", type=int, default=None, help="For this job, start from this line (if None, then 0)")
parser.add_argument("--job_end_idx", type=int, default=None, help="For this job, end at this line (if None, then till the end)")
parser.add_argument("--split_file_by_job", action="store_true", help="If true, set job_start_idx and job_end_idx by job_id and num_jobs")
parser.add_argument("--num_jobs", type=int, default=None, help="Total number of jobs (to split the file)")
parser.add_argument("--source_type", type=str, help="Can be 'hf', 'hf_local', 'arrow', 'jsonl', 'mds', or any of the above + '_list'; \
                    when + '_list', source is a txt with each line as a file name, and job_id determines which file")
parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir for HF dataset")
parser.add_argument("--hf_split", type=str, default="train", help="The split to use for HF dataset")
parser.add_argument("--tokenizer", type=str, default="llama3_tokenizer.model", help="Llama-3's 'tokenizer.model' path")
parser.add_argument("--input_ids_field", type=str, default="input_ids", help="The tokenized data field")
parser.add_argument("--indices", type=str, default=None, help="Path to a numpy indices")


parser.add_argument("--domain", type=str, help="Domain name, will be saved as metadata")
parser.add_argument("--metadata_filter", type=str, default=None, help="A string format of dictionary; only the item with matched metadata will be kept")
parser.add_argument("--metadata_mapping", type=str, default=None, help="Use this field to map to different subset")
parser.add_argument("--metadata_mapping_list", type=str, default=None, help="A string of a list of all possible mapping names, e.g., slimpajama domains")

# Strategy
# pack: just put tokens in the buffer; if reaches the target length, then save the first part and keep the extra in the buffer
# pack_complete: same as pack, but discard the extra part when reaching the target length
# length_filter: only keep documents that are longer than the min target lengths.
#     * There are multiple target length folders. Save it to the longest possible one
#     * The extra is also saved to the longest possible one recurrently
#     * The shorter length subset is always packed to the longest target length
# length_filter_complete: same as above but discard the extra part
parser.add_argument("--strategy", type=str, help="Can be 'pack', 'pack_complete' or 'length_filter'")
parser.add_argument("--add_boseos", type=bool, default=True, help="Add bos eos token")
parser.add_argument("--target_lengths", type=str, help="A string of an array, e.g., [65536, 32768, 16384, 8192] for length_filter strategy or 65536 for pack strategy")
parser.add_argument("--instruction", action="store_true", help="Whether this is an instruction dataset")
parser.add_argument("--shuffle", action="store_true", help="Shuffle within the worker")
parser.add_argument("--sort_by_length", action="store_true", help="Sort by length (reverse) within the worker")

# For MeCo
parser.add_argument("--add_metadata", action="store_true", help="Prepend 'metadata' field to the sequence")
parser.add_argument("--add_url", action="store_true", help="Prepend 'URL' field to the sequence")
parser.add_argument("--use_fixed_url", type=str, help="Use a fixed ULR (for ablation purpose)", default=None)
parser.add_argument("--use_short_url", action="store_true", help="Only use the absolute domain part of the URL")
parser.add_argument("--use_url_domain", action="store_true", help="Only use the domain part of the URL")
parser.add_argument("--use_url_suffix", action="store_true", help="Only use the suffix part of the URL")
parser.add_argument("--add_metadata_mask", action="store_true", help="Add masks for the metadata part")
parser.add_argument("--add_metadata_prefix", type=str, default="", help="Prefix for the metadata")
parser.add_argument("--add_metadata_suffix", type=str, default="\n\n", help="Suffix for the metadata")
parser.add_argument("--use_url_filtering", type=str, default=None, help="Use the specified JSON file to filter URLs")
parser.add_argument("--use_uuid", action="store_true", help="Use uuid to replace the URL (only work when use_url_filtering is not None)")


args = parser.parse_args()

assert args.strategy in ["pack", "pack_complete", "length_filter", "length_filter_complete", "bfd", "bfd_complete"]

if args.slurm_array_as_job_id:
    # Read env variable
    args.job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    print(f"Set job id to be {args.job_id}")

# Load llama3 tokenizer
print("Loading tokenizer...")
tok = Tokenizer(args.tokenizer)
print("Done")

# Prefix and suffix
# metadata_prefix = np.array([], dtype=np.uint32) if len(args.add_metadata_prefix) == 0 else np.array(tok.encode(args.add_metadata_prefix, bos=False, eos=False), dtype=np.uint32)
# metadata_suffix = np.array([], dtype=np.uint32) if len(args.add_metadata_suffix) == 0 else np.array(tok.encode(args.add_metadata_suffix, bos=False, eos=False), dtype=np.uint32)
metadata_prefix = args.add_metadata_prefix.replace("\\n", "\n")
metadata_suffix = args.add_metadata_suffix.replace("\\n", "\n")

# Load source text file
if args.source_type[-5:] == "_list":
    print(f"Reading the file list {args.source}")
    source_list = open(args.source).readlines()
    args.source = source_list[args.job_id].strip()
    print(f"Take the {args.job_id} file from the file list")
    args.source_type = args.source_type[:-5]

print(f"Source raw text data: {args.source}")
print("Loading source dataset...")
source = load_source(args.source_type, args.source, hf_split=args.hf_split, cache_dir=args.cache_dir)
print(f"Done. Loaded {len(source)} documents")

if args.indices is not None:
    index_order = np.load(args.indices)
    ntotal = len(index_order)
else:
    index_order = None
    ntotal = len(source)

if args.job_start_idx is not None and args.job_end_idx is not None:
    pass
elif args.split_file_by_job:
    assert args.num_jobs is not None
    per_job = ntotal // args.num_jobs
    args.job_start_idx = per_job * args.job_id
    args.job_end_idx = min(per_job * (args.job_id+1), ntotal)
else:
    args.job_start_idx = 0
    args.job_end_idx = ntotal

this_job_total = args.job_end_idx - args.job_start_idx
print(f"This job has job id {args.job_id}, and it uses line {args.job_start_idx} to {args.job_end_idx}, in total {this_job_total}")

# Set up multiprocessing
root_dir = args.target
per_worker = this_job_total // args.num_workers
columns = {"input_ids": "ndarray:uint32", "length": "uint64", "domain": "str", "indices": "ndarray:uint32"}
if args.instruction or args.add_metadata_mask:
    columns["mask"] = "ndarray:uint8"

target_lengths = eval(args.target_lengths)
# bfd: Best-fit-decreasing algorithm from "Fewer Truncations Improve Language Modeling"
if "pack" in args.strategy or "bfd" in args.strategy:
    assert isinstance(target_lengths, int)
    max_target_length = target_lengths
else:
    assert isinstance(target_lengths, list)
    target_lengths.sort(reverse=True)
    max_target_length = target_lengths[0]
metadata_filter = eval(args.metadata_filter) if args.metadata_filter is not None else None
metadata_mapping = args.metadata_mapping
metadata_mapping_list = eval(args.metadata_mapping_list) if args.metadata_mapping_list is not None else None

if metadata_mapping is not None:
    assert metadata_mapping_list is not None

def close_everything(out):
    if not isinstance(out, MDSWriter):
        for k, o in out.items():
            close_everything(o)
    else:
        out.finish()

def get_indices(list_of_seqs, max_len):
    indices = []
    start = 0
    for seq in list_of_seqs:
        indices.append((start, min(start+len(seq), max_len)))
        start += len(seq)
        if start >= max_len:
            break
    return np.array(indices, dtype=np.uint32)

from bfd import SingleBuffer, BFDBuffer
def write(info):
    source, process_id, start_idx, end_idx, index_order = info

    print(f"Process {process_id} start")

    if args.use_url_filtering is not None:
        print(f"Loading url filtering at {args.use_url_filtering}")
        url_mapping = json.load(open(args.use_url_filtering))
        url_total_documents = 0
        url_kept_documents = 0

    pack = "pack" in args.strategy
    bfd = "bfd" in args.strategy
    complete = "complete" in args.strategy
    add_boseos = args.add_boseos
    bos = tok.bos_id
    eos = tok.eos_id

    if pack or bfd:
        buffer_cls = BFDBuffer if bfd else SingleBuffer
        if metadata_mapping is not None:
            out = {
                f: MDSWriter(columns=columns, out=os.path.join(args.target, f, f"{args.job_id}-{process_id}"), compression=None)
                for f in metadata_mapping_list
            }
            buffers = {f: buffer_cls(out[f], max_target_length, complete=complete) for f in metadata_mapping_list}
        else:
            out = MDSWriter(columns=columns, out=os.path.join(args.target, f"{args.job_id}-{process_id}"), compression=None)
            buffers = buffer_cls(out, max_target_length, complete=complete)
    else:
        if metadata_mapping is not None:
            buffers = {f: {x: [] for x in target_lengths} for f in metadata_mapping_list}
            out = {
                f: {
                    x: MDSWriter(columns=columns, out=os.path.join(args.target, f, f"{x}-{max_target_length}", f"{args.job_id}-{process_id}"), compression=None)
                    for x in target_lengths
                }
                for f in metadata_mapping_list
            }

        else:
            buffers = {x: [] for x in target_lengths}
            out = {
                x: MDSWriter(columns=columns, out=os.path.join(args.target, f"{x}-{max_target_length}", f"{args.job_id}-{process_id}"), compression=None)
                for x in target_lengths
            }

    if index_order is None:
        index_order = range(start_idx, end_idx)
    if args.shuffle:
        index_order = list(index_order)
        random.shuffle(index_order)
    if args.sort_by_length:
        index_order = list(index_order)
        index_order.sort(key=lambda x: len(source[x]['input_ids']), reverse=True)

    for idx in tqdm(index_order, disable=process_id!=0):
        item = source[idx]

        try:
            # First filtering
            if metadata_filter is not None:
                skip = False
                for k, v in metadata_filter.items():
                    if item[k] != v:
                        # Skip this item
                        skip = True
                        break
                if skip:
                    continue

            # Then identify the mapped subset
            if metadata_mapping is not None:
                if item[metadata_mapping] not in metadata_mapping_list:
                    # Skip
                    print(f"Try to map to subset but did not find {item[metadata_mapping]} in {metadata_mapping_list} (key={metadata_mapping})")
                    continue
                current_out = out[item[metadata_mapping]]
                current_buffer = buffers[item[metadata_mapping]]
                domain = f"{args.domain}/{item[metadata_mapping]}"
            else:
                current_out = out
                current_buffer = buffers
                domain = args.domain

            # Concatenate bos/eos
            mask = None
            if args.add_metadata or args.add_url:
                if args.add_url:
                    if args.use_fixed_url is not None:
                        url = args.use_fixed_url
                    else:
                        url = item["url"]
                        if args.use_short_url:
                            parsed_url = re.sub(r'https?://', '', url)  # Remove http:// or https://
                            url = parsed_url.split('/')[0]  # Take only the domain part
                        if args.use_url_domain:
                            ext = tldextract.extract(url)
                            url = f"{ext.domain}.{ext.suffix}"
                        if args.use_url_suffix:
                            ext = tldextract.extract(url)
                            url = ext.suffix
                    
                    if args.use_url_filtering is not None:
                        if url not in url_mapping:
                            url = "unknown"
                        else:
                            url_kept_documents += 1
                            if args.use_uuid:
                                url = url_mapping[url]
                        url_total_documents += 1

                    encoded_metadata = np.array(tok.encode(metadata_prefix+url+metadata_suffix, bos=False, eos=False), dtype=np.uint32)
                else:
                    encoded_metadata = np.array(tok.encode(metadata_prefix+tok.decode(item["metadata"])+metadata_suffix, bos=False, eos=False), dtype=np.uint32)
                if add_boseos:
                    input_ids = np.concatenate([
                        np.array([bos], dtype=np.uint32),
                        encoded_metadata,
                        item["input_ids"],
                        np.array([eos], dtype=np.uint32)
                    ], 0)
                    if args.add_metadata_mask:
                        mask = np.ones(len(input_ids), dtype=np.uint8) 
                        mask[:1+len(encoded_metadata)] = 0
                else:
                    input_ids = np.concatenate([
                        encoded_metadata,
                        item["input_ids"],
                    ], 0)
                    if args.add_metadata_mask:
                        mask = np.ones(len(input_ids), dtype=np.uint8) 
                        mask[:len(encoded_metadata)] = 0
            else:
                if add_boseos:
                    input_ids = np.concatenate([
                        np.array([bos], dtype=np.uint32),
                        item["input_ids"],
                        np.array([eos], dtype=np.uint32)
                    ], 0)
                    if args.instruction:
                        mask = np.concatenate([
                            np.array([0], dtype=np.uint8),
                            item['mask'],
                            np.array([0], dtype=np.uint8)
                        ], 0)
                else:
                    input_ids = item['input_ids']
                    if args.instruction:
                        mask = item['mask']

            # Pack or bfd
            if pack or bfd:
                current_buffer.add(input_ids, mask, domain)
            else:
                # Recurrently process input_ids to the corresponding lengths
                while True:
                    # From longest to shortest
                    is_ok = False
                    for l in target_lengths:
                        if len(input_ids) >= l:
                            # if not the max target length, we force adding bos/eos
                            # because of the packing
                            if add_boseos and l != max_target_length:
                                input_ids[0] = bos
                                input_ids[l-1] = eos

                            # Add to the pack buffer
                            current_buffer[l].append(input_ids[:l])

                            if complete:
                                input_ids = input_ids[-1:]
                            else:
                                input_ids = input_ids[l:]

                            # Reached the pack length
                            if sum(len(b) for b in current_buffer[l]) >= max_target_length:
                                current_out[l].write({
                                    "input_ids": np.concatenate(current_buffer[l], 0),
                                    "domain": f"{domain}/{l}-{max_target_length}",
                                    "length": max_target_length,
                                    "indices": get_indices(current_buffer[l], max_target_length)
                                })
                                current_buffer[l].clear()

                            is_ok = True
                            break

                    if not is_ok:
                        # Shorter than the shortest length
                        break

        except Exception as e:
            print(f"Error: {e}")
            raise e

    close_everything(out)
    print(f"Process {process_id} finished successfully")
    if isinstance(buffers, dict):
        total_c_ctx_tokens = sum(b.get_corrupt_ctx_tokens() for b in buffers.values())
    else:
        total_c_ctx_tokens = buffers.get_corrupt_ctx_tokens()
    print(f"Total #tokens that have corrupted context: {total_c_ctx_tokens}.")
    if args.use_url_filtering is not None:
        print(f"Total #documents: {url_total_documents}, #kept documents: {url_kept_documents}")


try:
    os.makedirs(args.target, exist_ok=True)
except:
    pass

# Single process version; for debug and code understanding
if args.num_workers == 1:
    write((source, 0, args.job_start_idx, args.job_end_idx, None if index_order is None else index_order[args.job_start_idx:args.job_end_idx]))
else:
    args_group = []
    for i in range(args.num_workers):
        start_idx = args.job_start_idx + per_worker * i
        end_idx = args.job_start_idx + min(per_worker * (i+1), this_job_total)
        args_group.append((source, i, start_idx, end_idx, None if index_order is None else index_order[start_idx:end_idx]))

    with Pool(processes=args.num_workers) as pool:
        pool.map(write, args_group)

print('Finished')
