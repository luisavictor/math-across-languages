# model="Qwen/Qwen3-4B-Instruct-2507"
# echo "Running scaling/pruning for model: $model";


# ## German RACE (scale)
# CUDA_VISIBLE_DEVICES=5,6,3 python3 MathNeuro/MathNeuro.py \
#     --model $model \
#     --save_path results_missing/scale/results_gsm8k_race_german \
#     --train_dataset custom_datasets/GSM8k_German/gsm8k_de_train.csv \
#     --train_lm_eval_task gsm8k_de_cot \
#     --eval_datasets race_de \
#     --calibration_datasets custom_datasets/Race_German/race_de_train.csv \
#     --calibration_dataset_names Race \
#     --eval_dataset_subset 200 \
#     --num_samples 500 \
#     --random_state 1 \
#     --scalar 1.01 \
#     --batch_size "auto" \
#     --max_batch_size 16 \
#     --num_repeats 3 \
#     --store_params ;

# # ## English RACE (scale)
# CUDA_VISIBLE_DEVICES=5,6,3 python3 MathNeuro/MathNeuro.py \
#     --model $model \
#     --save_path results_missing/scale/results_gsm8k_race \
#     --train_dataset MathNeuro/data/gsm8k.csv \
#     --train_lm_eval_task gsm8k_cot \
#     --eval_datasets race \
#     --calibration_datasets MathNeuro/data/race.csv \
#     --calibration_dataset_names Race \
#     --eval_dataset_subset 200 \
#     --num_samples 500 \
#     --random_state 1 \
#     --scalar 1.01 \
#     --batch_size "auto" \
#     --max_batch_size 16 \
#     --num_repeats 3 \
#     --store_params ;


# # ## English MMLU (scale)
# CUDA_VISIBLE_DEVICES=5,6,3 python3 MathNeuro/MathNeuro.py \
#     --model $model \
#     --save_path results_missing/scale/results_gsm8k_mmlu \
#     --train_dataset MathNeuro/data/gsm8k.csv \
#     --train_lm_eval_task gsm8k_cot \
#     --eval_datasets mmlu \
#     --calibration_datasets MathNeuro/data/mmlu.csv \
#     --calibration_dataset_names MMLU \
#     --eval_dataset_subset 200 \
#     --num_samples 500 \
#     --random_state 1 \
#     --scalar 1.01 \
#     --batch_size "auto" \
#     --max_batch_size 16 \
#     --num_repeats 3 \
#     --store_params ;





model="meta-llama/Llama-3.1-8B-Instruct"

# ## German RACE (prune)
CUDA_VISIBLE_DEVICES=6,7 python3 MathNeuro/MathNeuro.py \
    --model $model \
    --save_path results/prune/results_gsm8k_race_german \
    --train_dataset custom_datasets/GSM8k_German/gsm8k_de_train.csv \
    --train_lm_eval_task gsm8k_de_cot \
    --eval_datasets race_de \
    --calibration_datasets custom_datasets/Race_German/race_de_train.csv \
    --calibration_dataset_names Race \
    --eval_dataset_subset 200 \
    --num_samples 500 \
    --random_state 1 \
    --batch_size "1" \
    --max_batch_size 32 \
    --num_repeats 3 \
    --store_params ;