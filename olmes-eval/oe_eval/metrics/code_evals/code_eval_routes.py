import logging
import multiprocessing
import os
from typing import List, Optional

logger = logging.getLogger()


def code_eval_local(
    batched_code_test: List[dict],
    timeout: float = 3.0,
    n_exe_workers: Optional[int] = None,
) -> List[dict]:
    """
    This function is used to test the generated code within a local sandbox.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from oe_eval.metrics.code_evals.code_execution_local import check_correctness

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if n_exe_workers is None:
        n_exe_workers = max(1, multiprocessing.cpu_count() // 2)

    with ThreadPoolExecutor(max_workers=n_exe_workers) as executor:
        futures = []
        check_results = []
        for res in batched_code_test:
            args = (
                res["res_id"],
                res["completion"],
                res["test"],
                timeout,
            )
            futures.append(executor.submit(check_correctness, *args))
        logger.info("Running test suites...")
        for future in as_completed(futures):
            check_results.append(future.result())

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return check_results
