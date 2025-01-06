import argparse
from streaming import MDSWriter, LocalDataset
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
import os
from llama3_tokenizer import Tokenizer, ChatFormat
import json
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import sys
from mds_tools import load_source

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str) # , nargs="+") current does not support list
parser.add_argument("--hf_secondary_path", type=str, default=None) # , nargs="+") current does not support list
parser.add_argument("--target", type=str)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--job_id", type=int, default=0, help="Should pass $SLURM_ARRAY_TASK_ID if using SLURM array job")
parser.add_argument("--slurm_array_as_job_id", action="store_true", help="Use $SLURM_ARRAY_TASK_ID to override the job id")
parser.add_argument("--job_start_idx", type=int, default=None, help="For this job, start from this line (if None, then 0)")
parser.add_argument("--job_end_idx", type=int, default=None, help="For this job, end at this line (if None, then till the end)")
parser.add_argument("--split_file_by_job", action="store_true", help="If true, set job_start_idx and job_end_idx by job_id and num_jobs")
parser.add_argument("--split_list_by_job", action="store_true", help="If true, split the whole list into num_jobs and take the job_id-th group")
parser.add_argument("--num_jobs", type=int, default=None, help="Total number of jobs (to split the file)")
parser.add_argument("--source_type", type=str, help="Can be 'hf', 'hf_local', 'arrow', 'jsonl', 'mds', or any of the above + '_list'; \
                    when + '_list', source is a txt with each line as a file name, and job_id determines which file")
parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir for HF dataset")
parser.add_argument("--hf_split", type=str, default="train", help="The split to use for HF dataset")
parser.add_argument("--text_field", type=str, default="text", help="The json/hf dataset field that represents text")
parser.add_argument("--metadata", type=str, default=None, help="A string format of dictionary, encompassing fields to save as metadata")
parser.add_argument("--domain_field", type=str, help="Domain name field, will be saved as metadata")
parser.add_argument("--domain", type=str, help="Domain name, will be saved as metadata")
parser.add_argument("--tokenizer", type=str, default="llama3_tokenizer.model", help="Llama-3's 'tokenizer.model' path")
parser.add_argument("--truncate_bytes", type=int, default=10000000000, help="Truncate the text field by 10GB to avoid memory overflow")
parser.add_argument("--instruction", action="store_true", help="Whether this is an instruction dataset")

args = parser.parse_args()

if args.slurm_array_as_job_id:
    # Read env variable
    args.job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    print(f"Set job id to be {args.job_id}")

# Load llama3 tokenizer
print("Loading tokenizer...")
tok = Tokenizer(args.tokenizer)
chattok = ChatFormat(tok)
print("Done")

# Load source text file
if args.source_type[-5:] == "_list":
    print(f"Reading the file list {args.source}")
    source_list = open(args.source).readlines()
    if args.split_list_by_job:
        files_per_job = len(source_list) // args.num_jobs
        args.source = source_list[files_per_job * args.job_id: files_per_job * (args.job_id+1)]
        print(f"Split the file list and handle {len(args.source)} files for this job (job id is {args.job_id}, file id from {files_per_job * args.job_id} to {files_per_job * (args.job_id+1)})")
    else:
        args.source = source_list[args.job_id].strip()
        print(f"Take the {args.job_id} file from the file list")
    args.source_type = args.source_type[:-5]

if isinstance(args.source, list):
    print(f"Loading {len(args.source)} files...")
    source = []
    for file_path in tqdm(args.source):
        print(f"Loading source dataset from {file_path.strip()}...")
        source += load_source(args.source_type, file_path.strip(), hf_split=args.hf_split, cache_dir=args.cache_dir, secondary_path=args.hf_secondary_path)
else:
    print(f"Source raw text data: {args.source}")
    print("Loading source dataset...")
    source = load_source(args.source_type, args.source, hf_split=args.hf_split, cache_dir=args.cache_dir, secondary_path=args.hf_secondary_path)

print(f"Done. Loaded {len(source)} documents")

if args.job_start_idx is not None and args.job_end_idx is not None:
    pass
elif args.split_file_by_job:
    assert args.num_jobs is not None
    per_job = len(source) // args.num_jobs
    args.job_start_idx = per_job * args.job_id
    args.job_end_idx = min(per_job * (args.job_id+1), len(source))
else:
    args.job_start_idx = 0
    args.job_end_idx = len(source)

this_job_total = args.job_end_idx - args.job_start_idx
print(f"This job has job id {args.job_id}, and it uses line {args.job_start_idx} to {args.job_end_idx}, in total {this_job_total}")


# Set up multiprocessing
root_dir = args.target
per_worker = this_job_total // args.num_workers
columns = {"input_ids": "ndarray:uint32", "length": "uint32", "domain": "str"}
if args.metadata is not None:
    metadata = eval(args.metadata)
    columns.update({k.split("/")[-1]: v for k, v in metadata.items()})
else:
    metadata = None
if args.instruction:
    columns.update({"mask": "ndarray:uint8"})

def get_item(line):
    if args.source_type == "jsonl" or args.source_type == "jsonl.zstd":
        return json.loads(line)
    else:
        return line

def get_field(item, field):
    field = field.split("/")
    for f in field:
        item = item[f]
    return item

def write(info):
    source, process_id, start_idx, end_idx = info

    print(f"Process {process_id} start")
    out = MDSWriter(columns=columns, out=os.path.join(args.target, f"{args.job_id}-{process_id}"), compression=None)

    for idx in tqdm(range(start_idx, end_idx), disable=process_id!=0):
        line = source[idx]
        try:
            item = get_item(line)
            mds_item = {metafield.split("/")[-1]: get_field(item, metafield) for metafield in metadata} if metadata is not None else {}
            if args.instruction:
                # args.text_field should correspond to an array of messages, where
                # each message is {"role": str (user/assistant/system), "content": str}
                chat = item[args.text_field]
                # For chat, it should start with <|begin_of_text|> but we'll add them in packing
                tokens = [] 
                mask = []
                for message in chat:
                    newtokens = chattok.encode_message(message)
                    if message['role'] == 'assistant':
                        mask += [0] * 4 # <|start_header_id)|>assistant<|end_header_id|>\n\n, 4 tokens
                        mask += [1] * (len(newtokens) - 4) # content + <|eot_id|>
                    else:
                        mask += [0] * len(newtokens)
                    tokens += newtokens
                assert len(tokens) == len(mask)                
                mds_item.update({
                    "input_ids": np.array(tokens, dtype=np.uint32),
                    "length": len(tokens),
                    "domain": args.domain or item[args.domain_field],
                    "mask": np.array(mask, dtype=np.uint8)
                })
                out.write(mds_item)
            else:
                tokens = tok.encode(item[args.text_field][:args.truncate_bytes], bos=False, eos=False)
                mds_item.update({
                    "input_ids": np.array(tokens, dtype=np.uint32),
                    "length": len(tokens),
                    "domain": args.domain or item[args.domain_field],
                })
                out.write(mds_item)
        except Exception as e:
            print(f"Error: {e}")
            continue
    out.finish()
    print(f"Process {process_id} finished successfully")


try:
    os.makedirs(args.target, exist_ok=True)
except:
    pass

# Single process version; for debug and code understanding
if args.num_workers == 1:
    write((source, 0, args.job_start_idx, args.job_end_idx))
else:
    args_group = []
    for i in range(args.num_workers):
        start_idx = args.job_start_idx + per_worker * i
        end_idx = args.job_start_idx + min(per_worker * (i+1), this_job_total)
        args_group.append((source, i, start_idx, end_idx))

    with Pool(processes=args.num_workers) as pool:
        pool.map(write, args_group)

print('Finished')
