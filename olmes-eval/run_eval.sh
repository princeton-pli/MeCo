# require setting environment variables: 
#   model: path to the model
#   task: one in arc_easy arc_challenge csqa hellaswag openbookqa piqa socialiqa winogrande mmlu truthfulqa
#   eval_dirname: name of the evaluation directory (will be established under the model path; can be used to distinguish different eval settings, for example, default vs. cond_inference)
#   (optional) prefix: the prefix text for conditional inference

export OEEVAL_PREFIX=${prefix:-""}

if [ "$task" == "truthfulqa" ]; then
    suffix=""
else
    suffix="::olmes"
fi

olmes \
    --model $model \
    --task ${task}${suffix} \
    --output-dir $model/$eval_dirname/$task \
    --model-args dtype=bfloat16,add_bos_token=True \
    $@
