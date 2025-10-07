import json
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams


@dataclass
class ScriptArguments:
    """
    The arguments for sampling.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=8192,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "the ids of the end of sentence tokens"},
    )


def main():
    # Arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print("model_path", script_args.model_name_or_path)

    # set seed
    torch.manual_seed(script_args.seed)
    np.random.seed(script_args.seed)

    # Load model and tokenizer
    llm = LLM(
        model=script_args.model_name_or_path,
        tokenizer=script_args.model_name_or_path,
        dtype="bfloat16",
        max_model_len=script_args.max_input_length,
        load_format="auto",
        seed=42,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    # Define sampling params
    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        top_p=1.0,
        max_tokens=script_args.max_new_tokens,
        n=script_args.K,
        stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    )

    # Load dataset and define prompt
    ds = load_dataset(script_args.dataset_name_or_path, split="train")
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    # Process dataset to make prompts
    def make_prompt(example):
        question = example["problem"]
        question = question + " " + instruction_following
        prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        return {"prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)}

    ds = ds.map(make_prompt)
    data_size = len(ds["prompt"])

    ## Split the dataset equally among GPUs
    k, m = divmod(data_size, script_args.my_world_size)
    start = script_args.local_index * k + min(script_args.local_index, m)
    end = (script_args.local_index + 1) * k + min(script_args.local_index + 1, m)
    ds = ds.select(np.arange(start, end))

    print([start, end])
    print(ds, script_args.dataset_name_or_path)
    print(ds[0])

    # Generate samples
    prompts = ds["prompt"]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # Save generations
    gathered_data = []
    for i, output in enumerate(outputs):
        gathered_data.append(
            {
                "problem": ds[i]["problem"],
                "prompt": ds[i]["prompt"],
                "gt": ds[i]["answer"],
                "responses": [out.text for out in output.outputs],
            }
        )

    print("I collect ", len(gathered_data), "samples")

    # Save to file
    with open(script_args.output_dir + str(script_args.local_index) + ".json", "w", encoding="utf8") as f:
        for i in range(len(gathered_data)):
            json.dump(gathered_data[i], f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
