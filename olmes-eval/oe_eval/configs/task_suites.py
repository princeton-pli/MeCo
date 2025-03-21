from oe_eval.data.agi_eval_tasks import AGI_EVAL_ENGLISH_DATASETS
from oe_eval.data.bbh_tasks import BBH_TASKS
from oe_eval.data.deepmind_math_categories import DEEPMIND_MATH_CATEGORIES
from oe_eval.data.math_task_types import MATH_TASK_TYPES
from oe_eval.data.mmlu_pro_categories import MMLU_PRO_CATEGORIES
from oe_eval.data.mmlu_tasks import MMLU_SUBJECTS
from oe_eval.data.mt_eval_tasks import MT_EVAL_TASKS
from oe_eval.data.tydiqa_languages import TYDIQA_LANGUAGES
from oe_eval.data.zero_scrolls_tasks import ZERO_SCROLLS_TASKS

TASK_SUITE_CONFIGS: dict = {}

# Each task suite config is a dictionary with keys:
# * tasks: List of task (or other task suite) IDs
# * primary_metric (optional): The primary metric to use for this task suite ("macro", "micro" or "mc_or_rc")

OLMES_MC_TASKS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
    "mmlu",
]
for olmes_mc_task in OLMES_MC_TASKS:
    TASK_SUITE_CONFIGS[f"{olmes_mc_task}::olmes"] = {
        "tasks": [f"{olmes_mc_task}:mc::olmes", f"{olmes_mc_task}:rc::olmes"],
        "primary_metric": "mc_or_rc",
    }
# All initial 9 OLMES tasks (not MMLU)
TASK_SUITE_CONFIGS["core_9mcqa::olmes"] = {
    "tasks": [f"{olmes_mc_task}::olmes" for olmes_mc_task in OLMES_MC_TASKS[:9]],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["core_9mcqa:rc::olmes"] = {
    "tasks": [f"{olmes_mc_task}:rc::olmes" for olmes_mc_task in OLMES_MC_TASKS[:9]],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["olmo_2_generative::olmes"] = {
    "tasks": [
        "coqa::olmes",
        "squad::olmes",
        "jeopardy::olmes",
        "naturalqs::olmes",
        "drop::olmes",
    ]
}
TASK_SUITE_CONFIGS["olmo_2_heldout::olmes"] = {
    "tasks": [
        "agi_eval_english:1shot::olmes",
        "bbh:cot-v1::olmes",
        "gsm8k::olmes",
        "mmlu_pro:mc::none",
        "triviaqa::olmes",
    ]
}
TASK_SUITE_CONFIGS["initial_4reasoning::olmes"] = {
    "tasks": [
        "agi_eval_english::olmes",
        "bbh:cot::olmes",
        "gsm8k::olmes",
        "minerva_math::olmes",
    ]
}
TASK_SUITE_CONFIGS["main_suite::olmo1"] = {
    "tasks": [
        "arc_challenge::olmo1",
        "arc_easy::olmo1",
        "boolq::olmo1",
        "copa::olmo1",
        "hellaswag::olmo1",
        "openbookqa::olmo1",
        "piqa::olmo1",
        "sciq::olmo1",
        "winogrande::olmo1",
        "truthfulqa::olmo1",
        "gsm8k::olmo1",
    ]
}
TASK_SUITE_CONFIGS["agi_eval_english::olmes"] = {
    "tasks": [
        f"agi_eval_{task_type}::olmes"
        for task_type in AGI_EVAL_ENGLISH_DATASETS
        if task_type != "sat-en-without-passage"
    ],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["agi_eval_english:1shot::olmes"] = {
    "tasks": [
        f"agi_eval_{task_type}:1shot::olmes"
        for task_type in AGI_EVAL_ENGLISH_DATASETS
        if task_type != "sat-en-without-passage"
    ],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["agi_eval_english:cot::none"] = {
    "tasks": [
        f"agi_eval_{task_type}:cot::olmes"
        for task_type in AGI_EVAL_ENGLISH_DATASETS
        if task_type != "sat-en-without-passage"
    ],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["agi_eval_english:0shot_cot::tulu3"] = {
    "tasks": [
        f"agi_eval_{task_type}:0shot_cot::tulu3"
        for task_type in AGI_EVAL_ENGLISH_DATASETS
        if task_type != "sat-en-without-passage"
    ],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["bbh:qa::none"] = {
    "tasks": [f"bbh_{task_type}:qa::none" for task_type in BBH_TASKS],
    "primary_metric": "micro",
}
TASK_SUITE_CONFIGS["bbh:cot::olmes"] = {
    "tasks": [f"bbh_{task_type}:cot::olmes" for task_type in BBH_TASKS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["bbh:cot-v1::olmes"] = {
    "tasks": [f"bbh_{task_type}:cot-v1::olmes" for task_type in BBH_TASKS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["bbh:cot::tulu"] = {
    "tasks": [f"bbh_{sub}:cot::tulu" for sub in BBH_TASKS],
    "primary_metric": "micro",
}
TASK_SUITE_CONFIGS["bbh:cot-v1::tulu"] = {
    "tasks": [f"bbh_{sub}:cot-v1::tulu" for sub in BBH_TASKS],
    "primary_metric": "micro",
}
TASK_SUITE_CONFIGS["deepmind_math::tulu"] = {
    "tasks": [f"deepmind_math_{cat}::tulu" for cat in DEEPMIND_MATH_CATEGORIES],
    "primary_metric": "micro",
}
TASK_SUITE_CONFIGS["deepmind_math:0shot_cot::tulu3"] = {
    "tasks": [f"deepmind_math_{cat}:0shot_cot::tulu3" for cat in DEEPMIND_MATH_CATEGORIES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["deepmind_math:0shot_cot-v1::tulu3"] = {
    "tasks": [f"deepmind_math_{cat}:0shot_cot-v1::tulu3" for cat in DEEPMIND_MATH_CATEGORIES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["deepmind_math:0shot_cot-v2::tulu3"] = {
    "tasks": [f"deepmind_math_{cat}:0shot_cot-v2::tulu3" for cat in DEEPMIND_MATH_CATEGORIES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["deepmind_math:0shot_cot-v3::tulu3"] = {
    "tasks": [f"deepmind_math_{cat}:0shot_cot-v3::tulu3" for cat in DEEPMIND_MATH_CATEGORIES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["minerva_math::llama3"] = {
    "tasks": [f"minerva_math_{task_type}::llama3" for task_type in MATH_TASK_TYPES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["minerva_math::llama3.1"] = {
    "tasks": [f"minerva_math_{task_type}::llama3.1" for task_type in MATH_TASK_TYPES],
    "primary_metric": "micro",
}
TASK_SUITE_CONFIGS["minerva_math::olmes"] = {
    "tasks": [f"minerva_math_{task_type}::olmes" for task_type in MATH_TASK_TYPES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["minerva_math::tulu"] = {
    "tasks": [f"minerva_math_{task_type}::tulu" for task_type in MATH_TASK_TYPES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu_pro:mc::none"] = {
    "tasks": [f"mmlu_pro_{cat}:mc::none" for cat in MMLU_PRO_CATEGORIES],
    "primary_metric": "micro",
}
TASK_SUITE_CONFIGS["mmlu_pro:cot::none"] = {
    "tasks": [f"mmlu_pro_{cat}:cot::none" for cat in MMLU_PRO_CATEGORIES],
    "primary_metric": "micro",
}
TASK_SUITE_CONFIGS["mmlu_pro:cot::llama3.1"] = {
    "tasks": [f"mmlu_pro_{cat}:cot::llama3.1" for cat in MMLU_PRO_CATEGORIES],
    "primary_metric": "micro",
}
TASK_SUITE_CONFIGS["mmlu_pro:0shot_cot::tulu3"] = {
    "tasks": [f"mmlu_pro_{cat}:0shot_cot::tulu3" for cat in MMLU_PRO_CATEGORIES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu:mc::olmes"] = {
    "tasks": [f"mmlu_{sub}:mc::olmes" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu:mc::tulu"] = {
    "tasks": [f"mmlu_{sub}:mc::tulu" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu:cot::none"] = {
    "tasks": [f"mmlu_{sub}:cot::none" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu:cot::reasoning"] = {
    "tasks": [f"mmlu_{sub}:cot::reasoning" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu:cot::summarize"] = {
    "tasks": [f"mmlu_{sub}:cot::summarize" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu:cot::reasoning-self-judge"] = {
    "tasks": [f"mmlu_{sub}:cot::reasoning-self-judge" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu:cot_nouha"] = {
    "tasks": [f"mmlu_{sub}:cot_nouha" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu:0shot_cot::tulu3"] = {
    "tasks": [f"mmlu_{sub}:0shot_cot::tulu3" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu:rc::olmes"] = {
    "tasks": [f"mmlu_{sub}:rc::olmes" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["mmlu::olmo1"] = {
    "tasks": [f"mmlu_{sub}::olmo1" for sub in MMLU_SUBJECTS],
    "primary_metric": "micro",
}
TASK_SUITE_CONFIGS["mt_eval:gpt3.5::std"] = {
    "tasks": [f"mt_eval_{task_type}:gpt3.5::std" for task_type in MT_EVAL_TASKS],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["tydiqa"] = {
    "tasks": [f"tydiqa_{lang}" for lang in TYDIQA_LANGUAGES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["tydiqa::tulu"] = {
    "tasks": [f"tydiqa_{lang}::tulu" for lang in TYDIQA_LANGUAGES],
    "primary_metric": "macro",
}
TASK_SUITE_CONFIGS["zero_scrolls::tulu"] = {
    "tasks": [f"zero_scrolls_{task}::tulu" for task in ZERO_SCROLLS_TASKS],
}
TASK_SUITE_CONFIGS["tulu_3_dev"] = {
    "tasks": [
        "gsm8k::tulu",
        "drop::llama3",
        "minerva_math::tulu",
        "codex_humaneval::tulu",
        "codex_humanevalplus::tulu",
        "ifeval::tulu",
        "popqa::tulu",
        "mmlu:mc::tulu",
        "alpaca_eval_v2::tulu",
        "bbh:cot-v1::tulu",
        "truthfulqa::tulu",
    ]
}

TASK_SUITE_CONFIGS["tulu_3_unseen"] = {
    "tasks": [
        "agi_eval_english:0shot_cot::tulu3",
        "gpqa:0shot_cot::tulu3",
        "mmlu_pro:0shot_cot::tulu3",
        "deepmind_math:0shot_cot::tulu3",
        "bigcodebench_hard::tulu",
        # Secondary formulations to track:
        "gpqa:0shot_cot::llama3.1",
        "mmlu_pro:cot::llama3.1",
        "bigcodebench::tulu",
    ]
}
