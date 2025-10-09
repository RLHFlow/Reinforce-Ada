import json
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from datasets import load_dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    dataset_path: Optional[str] = field(
        default="/home/xiongwei/gshf/data/gen_data",
        metadata={"help": "the location of the dataset name or path"},
    )
    save_path: Optional[str] = field(
        default="record.txt",
        metadata={"help": "the location of the output file"},
    )
    pass_rate: Optional[float] = field(
        default=0.125,
        metadata={
            "help": "the pass rate threshold for prompt selection, 0.125 and 0.313 for hard and easy, respectively"
        },
    )


def main():
    # Arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load dataset
    ds = load_dataset("json", data_files=script_args.dataset_path, split="train")

    # Prompt selection
    selected_prompts = []
    for sample in ds:
        if np.mean(sample["scores"]) <= script_args.pass_rate and np.sum(sample["scores"]) > 0:
            selected_prompts.append(
                {
                    "problem": sample["prompt"],
                    "answer": sample["gt"],
                }
            )

    # Save to file
    with open(script_args.save_path, "w", encoding="utf8") as f:
        for i in range(len(selected_prompts)):
            json.dump(selected_prompts[i], f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
