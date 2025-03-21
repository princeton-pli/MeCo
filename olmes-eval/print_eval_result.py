import json
import sys
import os
import numpy as np

def get_score(root, path, eval_type):

    scores = []
    
    for task in ["mmlu", "arc_easy", "arc_challenge", "csqa", "hellaswag", "openbookqa", "piqa", "socialiqa", "winogrande", "truthfulqa"]:
        try:
            r = json.loads(open(os.path.join(root, path, eval_type, task, "metrics-all.jsonl")).readlines()[0])
    
            if task == "truthfulqa":
                scores.append(r["metrics"]["mc2"] * 100)
            else:
                scores.append(r["metrics"]["primary_score"] * 100)
        except:
            scores.append(0) 
    
    scores.append(np.mean(scores))
    return scores


if __name__ == "__main__":
    root = ""
    path = sys.argv[1]
    eval_type = sys.argv[2]

    scores = get_score(root, path, eval_type)
    print("\t".join(["%.2f" % (s) for s in scores]) + "\t" + path)
    
