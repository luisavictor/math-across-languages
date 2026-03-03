
proportions="0.000001 0.00001 0.0001 0.001 0.005 0.01 0.025 0.05 0.1 0.15"


(
    ## Qwen German RACE (scale)
    CUDA_VISIBLE_DEVICES=5 python3 MathNeuroFast.py \
        --model "Qwen/Qwen3-4B-Instruct-2507" \
        --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_german \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_German/gsm8k_de_train.csv \
        --train_lm_eval_task gsm8k_de_cot \
        --eval_datasets race_de \
        --calibration_datasets /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/Race_German/race_de_train.csv \
        --calibration_dataset_names Race \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --random_state 1 \
        --scalar 1.01 \
        --batch_size "5" \
        --proportion $proportions \
        --num_repeats 3 \
        --pre_train_eval ;
    sleep 5s;

    ## Llama Hindi MMLU (scale)
    CUDA_VISIBLE_DEVICES=5 python3 MathNeuroFast.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" \
        --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_hindi_max300 \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_Hindi/gsm8k_hi_train.csv \
        --train_lm_eval_task gsm8k_hi_cot_max300 \
        --eval_datasets mmlu_hi_max300 \
        --calibration_datasets /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/MMLU_Hindi/mmlu_hi_val.csv \
        --calibration_dataset_names MMLU \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --random_state 1 \
        --scalar 1.01 \
        --proportion $proportions \
        --num_repeats 3 \
        --batch_size "3" \
        --pre_train_eval 
) &

sleep 5s;

(
    ## Qwen German MMLU (scale)
    CUDA_VISIBLE_DEVICES=4 python3 MathNeuroFast.py \
        --model "Qwen/Qwen3-4B-Instruct-2507" \
        --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_german \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_German/gsm8k_de_train.csv \
        --train_lm_eval_task gsm8k_de_cot \
        --eval_datasets mmlu_de \
        --calibration_datasets /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/MMLU_German/mmlu_de_val.csv \
        --calibration_dataset_names MMLU \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --random_state 1 \
        --scalar 1.01 \
        --proportion $proportions \
        --num_repeats 3 \
        --batch_size "5" \
        --pre_train_eval ;
        
    sleep 5s;

    ## Llama German MMLU (scale)
    CUDA_VISIBLE_DEVICES=4 python3 MathNeuroFast.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" \
        --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_german \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_German/gsm8k_de_train.csv \
        --train_lm_eval_task gsm8k_de_cot \
        --eval_datasets mmlu_de \
        --calibration_datasets /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/MMLU_German/mmlu_de_val.csv \
        --calibration_dataset_names MMLU \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --random_state 1 \
        --scalar 1.01 \
        --proportion $proportions \
        --num_repeats 3 \
        --batch_size "3" \
        --pre_train_eval 
) &


(
    ## Llama Hindi MMLU (prune)
    CUDA_VISIBLE_DEVICES=2 python3 MathNeuroFast.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" \
        --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_hindi_max300 \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_Hindi/gsm8k_hi_train.csv \
        --train_lm_eval_task gsm8k_hi_cot_max300 \
        --eval_datasets mmlu_hi_max300 \
        --calibration_datasets /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/MMLU_Hindi/mmlu_hi_val.csv \
        --calibration_dataset_names MMLU \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --random_state 1 \
        --proportion $proportions \
        --num_repeats 3 \
        --batch_size "3" ;

    sleep 5s;

    ## Qwen German MMLU (prune)
    CUDA_VISIBLE_DEVICES=2 python3 MathNeuroFast.py \
        --model "Qwen/Qwen3-4B-Instruct-2507" \
        --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_german \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_German/gsm8k_de_train.csv \
        --train_lm_eval_task gsm8k_de_cot \
        --eval_datasets mmlu_de \
        --calibration_datasets /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/MMLU_German/mmlu_de_val.csv \
        --calibration_dataset_names MMLU \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --random_state 1 \
        --proportion $proportions \
        --num_repeats 3 \
        --batch_size "5" ;
) &

sleep 5s;

(
    ## Llama German MMLU (prune)
    CUDA_VISIBLE_DEVICES=0 python3 MathNeuroFast.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" \
        --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_german \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_German/gsm8k_de_train.csv \
        --train_lm_eval_task gsm8k_de_cot \
        --eval_datasets mmlu_de \
        --calibration_datasets /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/MMLU_German/mmlu_de_val.csv \
        --calibration_dataset_names MMLU \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --random_state 1 \
        --proportion $proportions \
        --num_repeats 3 \
        --batch_size "3" ;

    sleep 5s;

    ## Llama English MMLU (prune)
    CUDA_VISIBLE_DEVICES=0 python3 MathNeuroFast.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" \
        --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/gsm8k.csv \
        --train_lm_eval_task gsm8k_cot \
        --eval_datasets mmlu \
        --calibration_datasets /raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/mmlu_val.csv \
        --calibration_dataset_names MMLU \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --random_state 1 \
        --proportion $proportions \
        --num_repeats 3 \
        --batch_size "3" 
) &

## Qwen English MMLU (prune)
CUDA_VISIBLE_DEVICES=1 python3 MathNeuroFast.py \
    --model "Qwen/Qwen3-4B-Instruct-2507" \
    --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu \
    --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/gsm8k.csv \
    --train_lm_eval_task gsm8k_cot \
    --eval_datasets mmlu \
    --calibration_datasets /raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/mmlu_val.csv \
    --calibration_dataset_names MMLU \
    --eval_dataset_subset 200 \
    --num_samples 500 \
    --random_state 1 \
    --proportion $proportions \
    --num_repeats 3 \
    --batch_size "5" ;

wait