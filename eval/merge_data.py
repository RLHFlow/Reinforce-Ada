import os
import json
import random
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    base_path: Optional[str] = field(
        default="/home/xiongwei/gshf/data/gen_data",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    num_datasets: Optional[int] = field(
        default=8,
        metadata={"help": "the location of the output file"},
    )


def main():
    # Arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Merge files
    all_dirs = []
    for i in range(script_args.num_datasets):
        all_dirs.append(os.path.join(script_args.base_path, f"{i}.json"))

    gathered_data = []
    for my_dir in all_dirs:
        ds = load_dataset("json", data_files=my_dir, split="train")
        print(len(ds))
        for sample in ds:
            gathered_data.append(sample)

    random.shuffle(gathered_data)
    print("I collect ", len(gathered_data), "samples")

    # Save to file
    with open(script_args.output_dir, "w", encoding="utf8") as f:
        for i in range(len(gathered_data)):
            json.dump(gathered_data[i], f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
