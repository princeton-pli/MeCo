#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

# Preload SRU
if os.environ.get('SRU') == '1':
    from sru import SRU
    print("Load SRU module successfully!")

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from trainer import trainer_addon

from data import get_multiple_domain_dataset, DataCollator

import simple_parsing
import yaml
from simple_parsing import ArgumentParser

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class OurArguments(TrainingArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    run_config: str = field(
        default=None,
        metadata={"help": "Run config file"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    hf_flash_attn2: bool = field(
        default=False,
        metadata={
            "help": "Add flash attn using HF's FA2 support; will be overridden if using modeling_varlen_llama"
        }
    )
    pytorch_flash_attn1: bool = field(
        default=False,
        metadata={
            "help": "Add flash attn using pytorch's FA1 support; will be overridden if using modeling_varlen_llama"
        }
    )
    varlenattn_llama: bool = field(
        default=False,
        metadata={"help": "Use variable length attention Llama"}
    )

    # Dataset
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    streaming_dataset: bool = field(
        default=False, metadata={"help": "Whether to use mosaicml-streaming dataset or not."}
    )
    streaming_train_root: str = field(
        default=None, metadata={"help": "The root directory of the streaming training dataset."}
    )
    streaming_val_root: str = field(
        default=None, metadata={"help": "The root directory of the streaming validation dataset."}
    )
    streaming_domains: str = field(
        default=None, metadata={"help": "The domains/proportions of the streaming dataset. Should be a JSON string."}
    )
    streaming_remote: bool = field(
        default=False, metadata={"help": "Whether to use remote streaming dataset or not."}
    )
    domains_and_proportions_train: str = field(
        default=None, metadata={"help": "Domain and proportions for the streaming dataset"}
    )
    domains_and_proportions_val: str = field(
        default=None, metadata={"help": "Domain and proportions for the streaming dataset"}
    )
    one_to_many_ratio_train: int = field(
        default=None, metadata={"help": "Turn one long sequence into multiple shorter instance (for training)"}
    )
    one_to_many_ratio_val: int = field(
        default=None, metadata={"help": "Turn one long sequence into multiple shorter instance (for validation)"}
    )
    return_indices: bool = field(
        default=False,
        metadata={"help": "Return indices for variable length attention (FA2 required)"}
    )
    sort_by_length_steps: int = field(
        default=0,
        metadata={
            "help":
                "Sort sequences in batches by length every n steps. "
                "0 to disable, 1 to sort every step (recommended), which will not change training dynamics, "
                "Can get even better performance with values >1, but this might change optimization and "
                "care needs to be taken that it divides the save steps, because otherwise data might be lost or seen twice "
                "after restarting training from a checkpoint."
        }
    )

    # Training
    streaming_dataset_resume: bool = field(
        default=True, metadata={"help": "Use streaming dataset state dict (for resuming). Will set ignore_data_skip to be True to avoid using HF's trainer dataset resuming"}
    )
    min_lr_ratio: float = field(
        default=0.0, metadata={"help": "Dropping LR to this ratio in the end"}
    )


def main():

    parser = ArgumentParser()
    parser.add_argument("--run_config", type=str, default=None)
    _, __ = parser.parse_known_args()
    args: OurArguments = simple_parsing.parse(OurArguments, config_path=_.run_config)

    print(args)
    args.max_length = args.block_size
    args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.should_log:
        # The default of args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f"distributed training: {args.parallel_mode.value == 'distributed'}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(args.seed)


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if args.config_overrides is not None:
            logger.info(f"Overriding config: {args.config_overrides}")
            config.update_from_string(args.config_overrides)
            logger.info(f"New config: {config}")
    
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, **tokenizer_kwargs)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    model_kwargs = {}
    if args.hf_flash_attn2:
        assert not args.pytorch_flash_attn1 
        model_kwargs["attn_implementation"] = "flash_attention_2"
        config._attn_implementation_internal = "flash_attention_2"
    elif args.pytorch_flash_attn1:
        model_kwargs["attn_implementation"] = "sdpa"
        config._attn_implementation_internal = "sdpa"
    else:
        model_kwargs["attn_implementation"] = "eager"
        config._attn_implementation_internal = "eager"

    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    logger.info(f"Use torch dtype for model parameters: {torch_dtype}")

    
    if args.return_indices or args.varlenattn_llama:
        # Using our llama with variable length attention support
        # Force using Flash Attention 2
        from models.modeling_varlen_llama import LlamaForCausalLM
        model_cls = LlamaForCausalLM
        model_kwargs.pop("attn_implementation") # This llama impl only has flashattn2 anyway
        logger.warning("Use variable length attention (FA2) Llama")
    else:
        model_cls = AutoModelForCausalLM

    if args.model_name_or_path:
        model = model_cls.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            **model_kwargs,
        )
    else:
        if model_cls == AutoModelForCausalLM:
            model = model_cls.from_config(config, trust_remote_code=args.trust_remote_code, torch_dtype=torch_dtype, **model_kwargs)
        else:
            model = model_cls._from_config(config, torch_dtype=torch_dtype, **model_kwargs)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if args.streaming_dataset:
        train_dataset = get_multiple_domain_dataset(
            root_dir=args.streaming_train_root, 
            domains_and_proportions=args.domains_and_proportions_train, 
            shuffle=True, 
            remote=args.streaming_remote, 
            block_size=args.block_size, 
            tokenizer=tokenizer, 
            one_to_many_ratio=args.one_to_many_ratio_train, 
            batch_size=args.per_device_train_batch_size,
            return_indices=args.return_indices,
            sort_by_length_mega_batch=args.sort_by_length_steps *
                args.gradient_accumulation_steps *
                args.per_device_train_batch_size // max(args.dataloader_num_workers, 1)
        )
        eval_dataset = get_multiple_domain_dataset(
            root_dir=args.streaming_val_root, 
            domains_and_proportions=args.domains_and_proportions_val, 
            shuffle=False, 
            remote=args.streaming_remote, 
            block_size=args.block_size, 
            tokenizer=tokenizer, 
            one_to_many_ratio=args.one_to_many_ratio_val, 
            batch_size=args.per_device_eval_batch_size,
            return_indices=args.return_indices,
        )
    else:
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                token=args.token,
                streaming=args.streaming,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                    cache_dir=args.cache_dir,
                    token=args.token,
                    streaming=args.streaming,
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                    cache_dir=args.cache_dir,
                    token=args.token,
                    streaming=args.streaming,
                )
        else:
            data_files = {}
            dataset_args = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
            extension = (
                args.train_file.split(".")[-1]
                if args.train_file is not None
                else args.validation_file.split(".")[-1]
            )
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = args.keep_linebreaks
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=args.cache_dir,
                token=args.token,
                **dataset_args,
            )
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{args.validation_split_percentage}%]",
                    cache_dir=args.cache_dir,
                    token=args.token,
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{args.validation_split_percentage}%:]",
                    cache_dir=args.cache_dir,
                    token=args.token,
                    **dataset_args,
                )
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        with args.main_process_first(desc="dataset map tokenization"):
            if not args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        if args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, config.max_position_embeddings)
        else:
            if args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with args.main_process_first(desc="grouping texts together"):
            if not args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=not args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

        if args.do_train:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = lm_datasets["train"]
            if args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        if args.do_eval:
            if "validation" not in tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = lm_datasets["validation"]
            if args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

    data_collator = DataCollator(args)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=None,
    )
    trainer = trainer_addon(trainer, streaming_dataset=args.streaming_dataset)


    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            args.max_train_samples if args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": args.model_name_or_path, "tasks": "text-generation"}
    if args.dataset_name is not None:
        kwargs["dataset_tags"] = args.dataset_name
        if args.dataset_config_name is not None:
            kwargs["dataset_args"] = args.dataset_config_name
            kwargs["dataset"] = f"{args.dataset_name} {args.dataset_config_name}"
        else:
            kwargs["dataset"] = args.dataset_name


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
