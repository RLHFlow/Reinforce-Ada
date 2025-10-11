import json
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from datasets import load_dataset
from transformers import HfArgumentParser
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from qwen_evaluation.grader import math_equal
from qwen_evaluation.parser import extract_answer


@dataclass
class ScriptArguments:
    dataset_path: Optional[str] = field(
        default="/home/xiongwei/gshf/data/gen_data",
        metadata={"help": "the location of the dataset name or path"},
    )
    record_path: Optional[str] = field(
        default="record.txt",
        metadata={"help": "the location of the output file"},
    )
    save_score: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to save the score for each sample"},
    )


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score


def main():
    # Arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load dataset
    ds = load_dataset("json", data_files=script_args.dataset_path, split="train")

    # Compute scores
    is_minerva_math = "minerva_math" in script_args.dataset_path.lower()

    all_scores = []
    for i in range(len(ds)):
        tmp_scores = []
        all_responses = ds[i]["responses"]
        ground_truth = ds[i]["gt"]
        for response in all_responses:
            if is_minerva_math:
                score = math_equal(extract_answer(response, "minerva_math"), ground_truth)
            else:
                score = compute_score(response, ground_truth)
            tmp_scores.append(score)
        all_scores.append(tmp_scores)

    # Save the average score
    with open(script_args.record_path, "w") as f:
        rounded_scores = np.round(np.mean(all_scores), 4)
        f.write(script_args.dataset_path + " " + str(rounded_scores) + "\n")

    # Save the score for each sample
    if script_args.save_score:
        gathered_data = []
        for i, sample in enumerate(ds):
            sample["scores"] = all_scores[i]
            gathered_data.append(sample)

    with open(script_args.dataset_path.split(".jsonl")[0] + "_score.jsonl", "w", encoding="utf8") as f:
        for i in range(len(gathered_data)):
            json.dump(gathered_data[i], f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
