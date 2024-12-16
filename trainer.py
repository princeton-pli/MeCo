import transformers
import torch
from transformers import Trainer, TrainerCallback, PrinterCallback
import time
from streaming import StreamingDataLoader
import logging
from transformers.trainer import logger
from transformers.trainer_utils import enable_full_determinism, find_executable_batch_size, get_last_checkpoint
from transformers.utils import is_sagemaker_dp_enabled, is_sagemaker_mp_enabled
from typing import Callable, Optional, List, Union
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import warnings
import math
import huggingface_hub.utils as hf_hub_utils
from transformers.trainer_callback import TrainerState
import glob
from transformers.trainer_pt_utils import reissue_pt_warnings
from accelerate.utils import load_fsdp_optimizer
from transformers.trainer import _get_fsdp_ckpt_kwargs

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

# logger = logging.getLogger(__name__)

class LogCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_time_interval = 0
        self.current_step = 0
        self.is_training = False
        self.max_steps = -1
        self.first_step_of_run = 0


    def on_train_begin(self, args, state, control, **kwargs):
        args.logging_steps = 1
        args.logging_strategy = "steps"
        if state.is_local_process_zero:
            self.log_time_interval = getattr(args, "log_time_interval", 0)
            if self.log_time_interval > 0:
                logger.info(f"Using log_time_interval {self.log_time_interval} s. This will override logging_steps and logging_strategy.")
            self.is_training = True
            self.current_step = 0
            self.start_time = time.time()
            self.last_log_time = self.start_time
            self.max_steps = state.max_steps
            self.first_step_of_run = state.global_step
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            if self.is_training:
                current_time = time.time()
                time_diff = current_time - self.last_log_time
                force = logs.get("force", False)
                if time_diff > self.log_time_interval or self.current_step >= self.max_steps - 1 or force:
                    self.last_log_time = current_time
                    steps_completed = max(self.current_step, 1)
                    steps_since_first = max(1, self.current_step - self.first_step_of_run)
                    remaining_steps = self.max_steps - steps_completed
                    pct_completed = (steps_completed / self.max_steps) * 100
                    time_since_start = current_time - self.start_time
                    remaining_time = (time_since_start / steps_since_first) * remaining_steps
                    update = {'completed': f'{pct_completed:.2f}% ({steps_completed:_} / {self.max_steps:_})', 'remaining time': self.format_duration(remaining_time)}
                    if getattr(args, "max_length", None) is not None:
                        total_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size * args.max_length
                        throughput = total_train_batch_size * steps_since_first / time_since_start
                        update.update({"throughput": throughput})
                    logger.info(str({**logs, **update}))
            else:
                logger.info(str(logs))


    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.current_step = state.global_step


    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.is_training = False


    @staticmethod
    def format_duration(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{int(hours)}:{int(minutes):02}:{int(seconds):02}'


def min_lr_bound(current_step: int, wrapped_func: Callable[[float], float], min_lr_ratio: float, warmup_steps: int):
    if current_step < warmup_steps:
        return wrapped_func(current_step)
    return min_lr_ratio + wrapped_func(current_step) * (1.0 - min_lr_ratio)


def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
    """

    self.lr_scheduler = self._original_create_scheduler(num_training_steps, optimizer)

    if getattr(self.args, "min_lr_ratio", 0.0) != 0.0:
        if isinstance(self.lr_scheduler, LambdaLR):
            lr_lambdas = self.lr_scheduler.lr_lambdas
            new_lr_lambdas = [
                lr_lambda
                if lr_lambda is None or isinstance(lr_lambda, partial) and lr_lambda.func == min_lr_bound
                else
                partial(min_lr_bound,
                        wrapped_func=lr_lambda,
                        min_lr_ratio=self.args.min_lr_ratio,
                        warmup_steps=self.args.get_warmup_steps(num_training_steps))
                for lr_lambda in lr_lambdas
            ]

            self.lr_scheduler.lr_lambdas = new_lr_lambdas
        else:
            raise NotImplementedError("Only LambdaLR is supported for min_lr_ratio")

    return self.lr_scheduler


def get_train_dataloader_for_streaming(self):
    """
    Because streaming handles the distributed data parallel by itself, we don't need special data loader.
    The plainest data loader is enough.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers, 
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    # Streaming is iterable so no need to set sampler etc.

    # Instead of use accelerate to prepare the dataloader, we just return a plain dataloader
    self.train_dataloader = StreamingDataLoader(train_dataset, **dataloader_params)

    def _get_batch_size(cls, batch):
        # Because we changed how data loader works
        # the batch size count is not accurate, which affects data loading
        return self.args.per_device_train_batch_size

    self.train_dataloader._get_batch_size = _get_batch_size.__get__(self.train_dataloader, StreamingDataLoader)

    assert self.train_dataset.replication is None, "Currently the dataset resuming on replication is not tested!"

    return self.train_dataloader


def get_eval_dataloader_for_streaming(self, eval_dataset):
    """
    Because streaming handles the distributed data parallel by itself, we don't need special data loader.
    The plainest data loader is enough.
    """
    if eval_dataset is None and self.eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")
    eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    data_collator = self.data_collator
    data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

    dataloader_params = {
        "batch_size": self.args.eval_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    # Streaming is iterable so no need to set sampler etc.

    # Instead of use accelerate to prepare the dataloader, we just return a plain dataloader
    return StreamingDataLoader(eval_dataset, **dataloader_params) 


import signal
from subprocess import call
class SIGUSR1Callback(transformers.TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        # signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")
        self.trainer = trainer

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Stop signal received...")

    def on_substep_end(self, args, state, control, **kwargs):
        if self.signal_received:
            self.trainer._save_checkpoint(self.trainer.model, None) # Note that here _save_checkpoint does not actually use this, so we can just pass on any model
            # The reason we don't set should_save but instead directly save here
            # is that streaming may collapse after receiving the signal and it
            # would be too late to wait till the save function is called.
            # Same reason for why we handle the single in both on_substep_end 
            # and on_step_end, even though ideally we want to do on_step_end.
            # control.should_save = True
            control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            self.trainer._save_checkpoint(self.trainer.model, None)
            # control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)


import os
from streaming import StreamingDataset
import json
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch.distributed as dist

def _save_checkpoint(self, model, trial, metrics=None):
    # A wrapper around the original _save_checkpoint function to save streaming dataset state

    # Save model checkpoint
    self._original_save_checkpoint(model, trial, metrics=metrics)

    # Get the path
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
    run_dir = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)

    # Save streaming dataset state
    if isinstance(self.train_dataset, StreamingDataset) and (not dist.is_initialized() or dist.get_rank() == 0):
        dataset_state_dict = self.train_dataloader.state_dict()
        logger.warn(f"Save streaming dataset state: {dataset_state_dict}")
        json.dump(dataset_state_dict, open(os.path.join(output_dir, "streaming_dataset_state.json"), "w"))


def _load_optimizer_and_scheduler(self, checkpoint):
    # A wrapper around the original _load_optimizer_and_scheduler to resume dataloader

    # Call the original function
    # self._original_load_optimizer_and_scheduler(checkpoint)

    # Below is copied from the original _load_optimizer_and_scheduler
    # But allow only loading optimizer if the scheduler does not exist

    """If optimizer and scheduler states exist, load them."""
    if checkpoint is None:
        return

    checkpoint_file_exists = (
        glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
        if is_sagemaker_mp_enabled()
        else (
            os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
            or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME_BIN))
            or (
                os.path.isdir(checkpoint)
                and any(
                    OPTIMIZER_NAME_BIN.split(".")[0] in folder_name
                    for folder_name in os.listdir(checkpoint)
                    if os.path.isdir(os.path.join(checkpoint, folder_name))
                )
            )
        )
    )
    if checkpoint_file_exists:
        logger.warn(f"Load optimizer state from {checkpoint}")
        # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
        # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
        # likely to get OOM on CPU (since we load num_gpu times the optimizer state
        map_location = self.args.device if self.args.world_size > 1 else "cpu"
        if self.is_fsdp_enabled:
            load_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.optimizer,
                self.model,
                checkpoint,
                **_get_fsdp_ckpt_kwargs(),
            )
        else:
            self.optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location)
            )

    if os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
        logger.warn(f"Load scheduler state from {checkpoint}")
        with warnings.catch_warnings(record=True) as caught_warnings:
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
        reissue_pt_warnings(caught_warnings)


    # Resume dataloader
    if self.args.streaming_dataset_resume:
        try:
            dataset_state_dict = json.load(open(os.path.join(checkpoint, "streaming_dataset_state.json")))
        except:
            logger.warn(f"Failed to load streaming dataset state from {checkpoint}")
            logger.warn(f"Will start from the beginning")
            self.args.ignore_data_skip = True
            # logger.warn(f"Fall back to the HF data skip")
            # self.args.ignore_data_skip = False

            return

        # First, disable HF's data skip 
        self.args.ignore_data_skip = True

        # Load the dataset state and reinit the dataloader
        logger.warn(f"Resume streaming dataset state from {checkpoint}: {dataset_state_dict}")
        self.train_dataloader.load_state_dict(dataset_state_dict)


def log(self, logs) -> None:
    # Replace the original log to include global steps

    if self.state.epoch is not None:
        logs["epoch"] = round(self.state.epoch, 2)
    if self.args.include_num_input_tokens_seen:
        logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
    if self.state.global_step is not None:
        logs["global_step"] = self.state.global_step

    self.state.log_history.append(logs)
    self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model

def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None

    outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        unwrapped_model = unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss


# Override the original train() to handle the case
# when resuming from a checkpoint but no trainer_state is there
# (e.g., continual training with optimizer states)
def train(
    self,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
    trial=None,
    ignore_keys_for_eval: Optional[List[str]] = None,
    **kwargs,
):
    """
    Main training entry point.

    Args:
        resume_from_checkpoint (`str` or `bool`, *optional*):
            If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
            `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
            of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
        trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
            The trial run or the hyperparameter dictionary for hyperparameter search.
        ignore_keys_for_eval (`List[str]`, *optional*)
            A list of keys in the output of your model (if it is a dictionary) that should be ignored when
            gathering predictions for evaluation during the training.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments used to hide deprecated arguments
    """
    if resume_from_checkpoint is False:
        resume_from_checkpoint = None

    # memory metrics - must set up as early as possible
    self._memory_tracker.start()

    args = self.args

    self.is_in_train = True

    # Attach NEFTune hooks if necessary
    if self.neftune_noise_alpha is not None:
        self.model = self._activate_neftune(self.model)

    # do_train is not a reliable argument, as it might not be set and .train() still called, so
    # the following is a workaround:
    if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
        self._move_model_to_device(self.model, args.device)

    if "model_path" in kwargs:
        resume_from_checkpoint = kwargs.pop("model_path")
        warnings.warn(
            "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
            "instead.",
            FutureWarning,
        )
    if len(kwargs) > 0:
        raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
    # This might change the seed so needs to run first.
    self._hp_search_setup(trial)
    self._train_batch_size = self.args.train_batch_size

    # Model re-init
    model_reloaded = False
    if self.model_init is not None:
        # Seed must be set before instantiating the model when using model_init.
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.model = self.call_model_init(trial)
        model_reloaded = True
        # Reinitializes optimizer and scheduler
        self.optimizer, self.lr_scheduler = None, None

    # Load potential model checkpoint
    if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        resume_from_checkpoint = get_last_checkpoint(args.output_dir)
        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

    if resume_from_checkpoint is not None:
        if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
            self._load_from_checkpoint(resume_from_checkpoint)
        # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
        # Edit from transformers: allow TRAINER_STATE_NAME to be missing
        if os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

    # If model was re-initialized, put it on the right device and update self.model_wrapped
    if model_reloaded:
        if self.place_model_on_device:
            self._move_model_to_device(self.model, args.device)
        self.model_wrapped = self.model

    inner_training_loop = find_executable_batch_size(
        self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
    )
    if args.push_to_hub:
        try:
            # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
            hf_hub_utils.disable_progress_bars()
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        finally:
            hf_hub_utils.enable_progress_bars()
    else:
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )


def trainer_addon(trainer, streaming_dataset=False):
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LogCallback)

    if streaming_dataset:
        trainer.get_train_dataloader = get_train_dataloader_for_streaming.__get__(trainer, Trainer)
        trainer.get_eval_dataloader = get_eval_dataloader_for_streaming.__get__(trainer, Trainer)

    trainer.add_callback(SIGUSR1Callback(trainer))

    trainer._original_save_checkpoint = trainer._save_checkpoint
    trainer._save_checkpoint = _save_checkpoint.__get__(trainer, Trainer)

    trainer._original_load_optimizer_and_scheduler = trainer._load_optimizer_and_scheduler
    trainer._load_optimizer_and_scheduler = _load_optimizer_and_scheduler.__get__(trainer, Trainer)

    trainer.log = log.__get__(trainer, Trainer)
    trainer.compute_loss = compute_loss.__get__(trainer, Trainer)

    trainer._original_create_scheduler = trainer.create_scheduler
    trainer.create_scheduler = create_scheduler.__get__(trainer, Trainer)

    trainer._original_train = trainer.train
    trainer.train = train.__get__(trainer, Trainer)

    return trainer
