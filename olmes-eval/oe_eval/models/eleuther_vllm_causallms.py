import copy
from importlib.util import find_spec
from typing import List, Optional, Tuple, cast

import torch
import transformers
from lm_eval.models.utils import Collator
from lm_eval.models.vllm_causallms import VLLM
from tqdm import tqdm

from oe_eval.components.requests import GenerateUntilRequest, LoglikelihoodRequest
from oe_eval.utils import cut_at_stop_sequence

# Verbose versions of key methods in VLLM from lm_eval.models.vllm_causallms


class VLLM_Verbose(VLLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM  # for encode_pairs to work

    def __init__(
        self,
        pretrained="gpt2",
        device: Optional[str] = None,
        # dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        # revision: Optional[str] = None,
        # trust_remote_code: Optional[bool] = False,
        # tokenizer: Optional[str] = None,
        # tokenizer_mode: Literal["auto", "slow"] = "auto",
        # tokenizer_revision: Optional[str] = None,
        # add_bos_token: Optional[bool] = False,
        # tensor_parallel_size: int = 1,
        # quantization: Optional[str] = None,
        # max_gen_toks: int = 256,
        # swap_space: int = 4,
        # batch_size: Union[str, int] = 1,
        # max_batch_size=None,
        # max_length: int = None,
        # max_model_len: int = None,
        # seed: int = 1234,
        # gpu_memory_utilization: float = 0.9,
        # data_parallel_size: int = 1,
        **kwargs,
    ):
        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                "model-type is `vllm` but `vllm` is not installed! Please install vllm via `pip install -e .[gpu]`"
            )
        if device is None:
            device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
        if torch.cuda.device_count() > 1:
            kwargs["tensor_parallel_size"] = torch.cuda.device_count()
        # if "revision" in kwargs and kwargs["revision"] is None:
        # Hack to deal with Eleuther using "main" as a default for "revision", not needed for VLLM
        #    kwargs = kwargs.copy()
        #    del kwargs["revision"]
        super().__init__(pretrained, device=device, **kwargs)

    def generate_until_verbose(
        self, requests: List[GenerateUntilRequest], disable_tqdm: bool = False
    ) -> List[str]:
        # oe-eval: Convert compute context and all_gen_kwargs in a custom way, then minimal changes needed below
        list_context = []
        list_gen_kwargs = []
        for request in requests:
            kwargs = request.generation_kwargs
            if kwargs is None:
                kwargs = {}
            else:
                kwargs = kwargs.copy()
            kwargs["until"] = request.stop_sequences
            list_context.append(cast(str, request.context))
            list_gen_kwargs.append(kwargs)
        res = []

        # batch tokenize contexts
        # context, all_gen_kwargs = zip(*(req.arg for req in requests))
        list_context_encoding = self.tokenizer(list_context, add_special_tokens=False).input_ids
        requests_tup = [
            ((a, b), c) for a, b, c in zip(list_context, list_context_encoding, list_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests_tup, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests_tup),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}")
            # add EOS token to stop sequences
            eos = self.tokenizer.decode(self.eot_token_id)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            truncate_context = kwargs.pop("truncate_context", True)

            if truncate_context:
                # set the max length in tokens of inputs ("context_enc")
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                if max_ctx_len <= 0:
                    raise ValueError(
                        f"max_gen_toks ({max_gen_toks}) is greater than max_length ({self.max_length}), "
                        + '"consider using "truncate_context": False'
                    )
            else:
                # If truncate_context is False we modify max_gen_toks instead, to keep full context
                max_ctx_len = self.max_length
                # The first context is the longest because of ordering above
                max_batch_toks = len(context_encoding[0])
                max_gen_toks = min(max_gen_toks, self.max_length - max_batch_toks)
                max_gen_toks = max(max_gen_toks, 0)
                if max_gen_toks == 0:
                    raise ValueError(
                        f"Effective max_gen_toks is 0 (max_batch_toks = {max_batch_toks}, max_ctx_len = {max_ctx_len}) "
                        + '"consider using "truncate_context": True'
                    )

            context_encoding_trunc = [x[-max_ctx_len:] for x in context_encoding]

            kwargs["logprobs"] = 1  # oe-eval: always return logprobs
            # perform batched generation
            cont = self._model_generate(
                requests=context_encoding_trunc,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )

            # cache generations
            for output, context in zip(cont, context):
                completion_output = output.outputs[0]
                generated_text_raw = completion_output.text
                generated_text = cut_at_stop_sequence(
                    generated_text_raw, until
                )  # oe_eval: catch stop sequence
                num_tokens = len(completion_output.token_ids)
                cumulative_logprob = completion_output.cumulative_logprob
                res1 = {
                    "continuation": generated_text,
                    "num_tokens": num_tokens,
                    # "tokens": completion_output.token_ids,
                }
                if cumulative_logprob is not None:
                    res1["sum_logits"] = cumulative_logprob
                    # log_probs = [list(p.values())[0].logprob for p in completion_output.logprobs]
                    # res1["logits"] = log_probs
                if generated_text_raw != generated_text:
                    res1["continuation_raw"] = generated_text_raw
                res.append(res1)  # oe_eval: change to dict here
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), generated_text)
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def loglikelihood_verbose(
        self,
        requests: List[LoglikelihoodRequest],
        override_bs: Optional[int] = None,
        disable_tqdm: bool = False,
    ) -> List[dict]:
        # This method is just copied from eleuther_huggingface for now
        new_reqs = []
        for req in requests:
            context = cast(str, req.context)
            continuation = req.continuation
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        return self._loglikelihood_tokens_verbose(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens_verbose(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        verbose_token_logits: bool = False,
    ) -> List[dict]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)

            for output, ctxlen, (cache_key, _, _), inp in zip(outputs, ctxlens, chunk, inputs):
                answer = self._parse_logprobs_verbose(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                    verbose_token_logits=verbose_token_logits,
                )

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    def _parse_logprobs_verbose(
        self, tokens: List, outputs, ctxlen: int, verbose_token_logits=False
    ) -> dict:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            verbose answer dict
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            # vLLM changed the return type of logprobs from float
            # to a Logprob object storing the float value + extra data
            # (https://github.com/vllm-project/vllm/pull/3065).
            # If we are dealing with vllm's Logprob object, return
            # the logprob value stored as an attribute. Otherwise,
            # return the object itself (which should be a float
            # for older versions of vLLM).
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            (
                {token: coerce_logprob_to_num(logprob) for token, logprob in logprob_dict.items()}
                if logprob_dict is not None
                else None
            )
            for logprob_dict in continuation_logprobs_dicts
        ]

        # Calculate continuation_logprobs
        # assume ctxlen always >= 1
        logits = [
            logprob_dict.get(token)
            for token, logprob_dict in zip(tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:])
        ]
        continuation_logprobs = sum(logits)

        # Determine if is_greedy
        is_greedy = True
        for token, logprob_dict in zip(tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]):
            # Get the token with the maximum log probability from the logprob_dict
            if logprob_dict:  # Ensure the logprob_dict is not None
                top_token = max(logprob_dict, key=logprob_dict.get)
                if top_token != token:
                    is_greedy = False
                    break

        verbose_answer: dict = {
            "sum_logits": continuation_logprobs,
            "num_tokens": len(logits),
            "num_tokens_all": len(tokens),
            "is_greedy": is_greedy,
        }
        if verbose_token_logits:
            instance_tokens = [
                self.tok_decode(x, skip_special_tokens=False) for x in tokens[ctxlen:]
            ]
            verbose_answer["tokens"] = instance_tokens
            verbose_answer["logits"] = logits

        return verbose_answer
