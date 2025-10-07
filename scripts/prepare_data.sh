#!/bin/bash

# Model used to select prompt
model_name=Qwen/Qwen2.5-Math-1.5B
output_dir="./data/openr1"
mkdir -p $output_dir

# Configuration
pass_rate=0.125
K=16
GPUS=(0 1 2 3 4 5 6 7)
ori_prompt_set="weqweasdas/from_default_filtered_openr1"
        
# Generate data in parallel
echo "Starting parallel data generation..."
for ((i=0; i<${#GPUS[@]}; i++)); do
    CUDA_VISIBLE_DEVICES=$i python3 eval/gen_data.py \
        --local_index ${i} \
        --my_world_size ${#GPUS[@]} \
        --model_name_or_path ${model_name} \
        --output_dir ${output_dir} \
        --K $K \
        --dataset_name_or_path ${ori_prompt_set} &
done

wait
echo "Data generation completed."
        
# Merge the generated data
echo "Merging data..."
python3 eval/merge_data.py \
    --base_path ${output_dir} \
    --output_dir ${output_dir}/merged_data.jsonl \
    --num_datasets ${#GPUS[@]}
        
# Compute scores
echo "Computing scores..."
python3 eval/compute_score.py \
    --dataset_path ${output_dir}/merged_data.jsonl \
    --record_path ${output_dir}/record.txt \
    --save_score True

# Select prompts
echo "Selecting prompts..."
python3 data_process/prompt_selection.py \
    --dataset_path ${output_dir}/merged_data.jsonl \
    --save_path ${output_dir}/selected_data_${pass_rate}.jsonl \
    --pass_rate ${pass_rate}

# Convert to verl training format
echo "Converting to verl training format..."
python3 data_process/reformat.py \
    --local_dir ${output_dir} \
    --model_name_or_path ${model_name} \
    --data_source ${output_dir}/selected_data_${pass_rate}.jsonl 

# Generate validation set
echo "Generating validation set..."
python3 data_process/get_validation_set.py \
    --local_dir ${output_dir} \
    --model_name_or_path ${model_name} 

echo "All data processing completed!"
