model=$1
echo "Running scaling/pruning/finetuning for model: $model";

## German RACE (scale)
CUDA_VISIBLE_DEVICES=0 python3 MathNeuro/MathNeuro.py \
    --model $model \
    --save_path results/scale/results_gsm8k_race_german \
    --train_dataset custom_datasets/GSM8k_German/gsm8k_de_train.csv \
    --train_lm_eval_task gsm8k_de_cot \
    --eval_datasets race_de \
    --calibration_datasets custom_datasets/Race_German/race_de_train.csv \
    --calibration_dataset_names Race \
    --eval_dataset_subset 200 \
    --num_samples 500 \
    --random_state 1 \
    --pre_train_eval \
    --scalar 1.01 \
    --proportion 0.2 \
    --batch_size "auto" \
    --max_batch_size 32 \
    --num_repeats 1 \
    --store_params ;

# ## English RACE (scale)
CUDA_VISIBLE_DEVICES=0 python3 MathNeuro/MathNeuro.py \
    --model $model \
    --save_path results/scale/results_gsm8k_race \
    --train_dataset MathNeuro/data/gsm8k.csv \
    --train_lm_eval_task gsm8k_cot \
    --eval_datasets race \
    --calibration_datasets MathNeuro/data/race.csv \
    --calibration_dataset_names Race \
    --eval_dataset_subset 200 \
    --num_samples 500 \
    --random_state 1 \
    --pre_train_eval \
    --scalar 1.01 \
    --proportion 0.2 \
    --batch_size "auto" \
    --max_batch_size 32 \
    --num_repeats 1 \
    --store_params ;