from typing import Any, Dict, List

import wandb


def wandb_log_metrics(wandb_run_path: str, metrics_all: List[Dict[str, Any]]):
    wandb_run = wandb.Api().run(wandb_run_path)
    for m in metrics_all:
        wandb_run.summary[f"oe_eval_metrics/{m['task_name']}"] = {
            **m["metrics"],
            "num_instances": m["num_instances"],
            "task_config": m["task_config"],
        }
    wandb_run.update()
    print(f"Logged metrics to {wandb_run.url}")
