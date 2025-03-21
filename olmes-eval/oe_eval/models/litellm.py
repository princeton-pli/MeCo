import logging
import re
from typing import List, Optional

from litellm import completion, get_supported_openai_params
from lm_eval import utils
from lm_eval.api.model import LM

from oe_eval.components.requests import GenerateUntilRequest
from oe_eval.utils import cut_at_stop_sequence

eval_logger = utils.eval_logger


class LiteLLM(LM):
    LLM_OPTIONS_RENAMES = {"max_gen_toks": "max_tokens"}
    LLM_OPTIONS_SKIP = ["truncate_context"]  # These are no-ops for LiteLLM

    def __init__(
        self,
        pretrained: str,  # Using this name for model name to match other model types for now
        batch_size: int = 1,
        max_length: int = 8192,
        api_base_url: Optional[str] = None,
        max_api_retries: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        self._max_length = max_length
        self.model = pretrained
        self.batch_size = batch_size
        self._api_base_url = api_base_url
        self._max_api_retries = max_api_retries
        if kwargs:
            eval_logger.warning(f"LiteLLM got ignores the following kwargs: {kwargs}")
        self.supported_openai_params = get_supported_openai_params(self.model)

    def max_length(self) -> int:
        return self._max_length

    def loglikelihood_verbose(self, **args):
        raise NotImplementedError("loglikelihood_verbose not implemented for LiteLLM")

    def generate_until(self, **args):
        raise NotImplementedError("generate_until not implemented for LiteLLM")

    def loglikelihood(self, **args):
        raise NotImplementedError("loglikelihood not implemented for LiteLLM")

    def loglikelihood_rolling(self, **args):
        raise NotImplementedError("loglikelihood_rolling not implemented for LiteLLM")

    def generate_until_verbose(
        self, requests: List[GenerateUntilRequest], disable_tqdm: bool = False
    ) -> List[dict]:
        # We ignore batch size, and will never truncate the context
        res = []
        eval_logger.info(f"Generating reponses for {len(requests)} requests")
        counter = 0
        for request in requests:
            counter += 1
            if counter % 100 == 0:
                eval_logger.info(f"Generated {counter}/{len(requests)} responses")
            kwargs = request.generation_kwargs
            if kwargs is None:
                kwargs = {}
            else:
                kwargs = kwargs.copy()
            stop_sequences = request.stop_sequences
            for key, value in self.LLM_OPTIONS_RENAMES.items():
                if key in kwargs:
                    kwargs[value] = kwargs.pop(key)
            for key in self.LLM_OPTIONS_SKIP:
                kwargs.pop(key, None)
            kwargs["num_retries"] = self._max_api_retries
            do_sample = kwargs.pop("do_sample", True)
            if not do_sample:
                kwargs["temperature"] = 0.0

            context = request.context
            messages: list = []
            assistant_prefix = None
            if isinstance(context, str):
                messages = [{"role": "user", "content": context}]
            elif isinstance(context, dict):
                messages = context["messages"]
                assistant_prefix = context.get("assistant_prefix")
                if assistant_prefix:
                    # assistant_prefix might not work as intended for certain models
                    messages.append({"role": "assistant", "content": assistant_prefix})
            else:
                raise ValueError(f"Unexpected context type: {type(context)}")
            # Disable logging for litellm call, it leaks API key for Gemini, e.g.
            logging.disable(logging.INFO)
            try:
                output_raw = completion(model=self.model, messages=messages, **kwargs)
                content = output_raw["choices"][0]["message"]["content"]
                if assistant_prefix:
                    # strip assistant prefix from start of reply if it's there, to be consistent
                    # with treatment for other models
                    content = re.sub(f"^\\s*{re.escape(assistant_prefix)}\\s*", "", content)
                if stop_sequences:
                    # We'll simply apply stop sequences after generation for simplicity
                    # E.g., Claude doesn't support whitespace-only like "\n\n" and other models
                    # only support up to 4 sequences
                    content = cut_at_stop_sequence(content, stop_sequences)
                if not isinstance(content, str):
                    # eval_logger.warning(f"Unexpected output for content: {content}")
                    content = ""
                res1 = {
                    "continuation": content,
                    "llm_response": output_raw.to_dict(),
                    "price": output_raw._hidden_params.get("response_cost", 0.0),
                    "num_tokens": output_raw["usage"]["completion_tokens"],
                    "num_tokens_total": output_raw["usage"]["total_tokens"],
                }
            except Exception as e:
                res1 = {
                    "continuation": "",
                    "price": 0,
                    "num_tokens": 0,
                    "num_tokens_total": 0,
                    "llm_error": str(e),
                }
            logging.disable(logging.NOTSET)
            res.append(res1)

        return res
