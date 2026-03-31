
proportions="0.001 0.01 0.1"
# random_prune_fractions="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
random_prune_fractions="0.05 0.1 0.2 0.4 0.6 0.8"

ENGLISH_LM_EVAL_TASK="gsm8k_cot"
ENGLISH_GSM8K_PATH="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/gsm8k.csv"
ENGLISH_RACE_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/race.csv"
ENGLISH_MMLU_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/mmlu_val.csv"


HINDI_LM_EVAL_TASK="gsm8k_hi_cot_max300"
HINDI_GSM8K_PATH="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_Hindi/gsm8k_hi_train.csv"
HINDI_RACE_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/Race_Hindi/race_hi_train.csv"
HINDI_MMLU_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/MMLU_Hindi/mmlu_hi_val.csv"

GERMAN_LM_EVAL_TASK="gsm8k_de_cot"
GERMAN_GSM8K_PATH="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_German/gsm8k_de_train.csv"
GERMAN_RACE_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/Race_German/race_de_train.csv"
GERMAN_MMLU_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/MMLU_German/mmlu_de_val.csv"

FRENCH_LM_EVAL_TASK="gsm8k_fr_cot"
FRENCH_GSM8K_PATH="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_French/gsm8k_fr_train.csv"
FRENCH_RACE_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/Race_French/race_fr_train.csv"
FRENCH_MMLU_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/MMLU_French/mmlu_fr_val.csv"

model=$1
if [[ $model == *"Qwen"* ]]; then
    export batch_size="18"
elif [[ $model == *"Llama-3.2-1B-Instruct"* ]]; then
    export batch_size="24"
else
    export batch_size="16"
fi


echo "Running scaling/pruning for model: $model";

scalar=0
save_root="/raid/s3/opengptx/behzad_shomali/LabTest/results/prune"

## English RACE 
# env CUDA_VISIBLE_DEVICES=1 python3 MathNeuroFast.py \
#     --model $model \
#     --save_path ${save_root}/results_gsm8k_race \
#     --train_dataset $ENGLISH_GSM8K_PATH \
#     --train_lm_eval_task $ENGLISH_LM_EVAL_TASK \
#     --eval_datasets race \
#     --calibration_datasets $ENGLISH_RACE_CALIBRATION_DATASET \
#     --calibration_dataset_names Race \
#     --eval_dataset_subset 200 \
#     --num_samples 500 \
#     --random_state 1 \
#     --scalar $scalar \
#     --batch_size $batch_size \
#     --proportion $proportions \
#     --num_repeats 3 \
#     --random_prune \
#     --random_prune_seed 42 123 456 \
#     --random_prune_fractions $random_prune_fractions &

# sleep 20s;

# (
    ## Hindi RACE 
    # env CUDA_VISIBLE_DEVICES=$2 python3 MathNeuroFast.py \
    #     --model $model \
    #     --save_path ${save_root}/results_gsm8k_race_hindi_max300 \
    #     --train_dataset $HINDI_GSM8K_PATH \
    #     --train_lm_eval_task $HINDI_LM_EVAL_TASK \
    #     --eval_datasets race_hi_max300 \
    #     --calibration_datasets $HINDI_RACE_CALIBRATION_DATASET \
    #     --calibration_dataset_names Race \
    #     --eval_dataset_subset 200 \
    #     --num_samples 500 \
    #     --random_state 1 \
    #     --scalar $scalar \
    #     --batch_size $batch_size \
    #     --proportion $proportions \
    #     --num_repeats 3 \
    #     --random_prune \
    #     --random_prune_seed 42 123 456 \
    #     --random_prune_fractions $random_prune_fractions ;
    
    # sleep 20s;
    # ## German RACE 
    # env CUDA_VISIBLE_DEVICES=3 python3 MathNeuroFast.py \
    #     --model $model \
    #     --save_path ${save_root}/results_gsm8k_race_german \
    #     --train_dataset $GERMAN_GSM8K_PATH \
    #     --train_lm_eval_task $GERMAN_LM_EVAL_TASK \
    #     --eval_datasets race_de \
    #     --calibration_datasets $GERMAN_RACE_CALIBRATION_DATASET \
    #     --calibration_dataset_names Race \
    #     --eval_dataset_subset 200 \
    #     --num_samples 500 \
    #     --random_state 1 \
    #     --scalar $scalar \
    #     --batch_size $batch_size \
    #     --proportion $proportions \
    #     --num_repeats 3 \
    #     --random_prune \
    #     --random_prune_seed 42 123 456 \
    #     --random_prune_fractions $random_prune_fractions ;
# ) &


# env CUDA_VISIBLE_DEVICES=$2 python3 MathNeuroFast.py \
#     --model $model \
#     --save_path ${save_root}/results_gsm8k_race_hindi_max300 \
#     --train_dataset $HINDI_GSM8K_PATH \
#     --train_lm_eval_task $HINDI_LM_EVAL_TASK \
#     --eval_datasets race_hi_max300 \
#     --calibration_datasets $HINDI_RACE_CALIBRATION_DATASET \
#     --calibration_dataset_names Race \
#     --eval_dataset_subset 200 \
#     --num_samples 500 \
#     --random_state 1 \
#     --scalar $scalar \
#     --batch_size $batch_size \
#     --proportion $proportions \
#     --num_repeats 3 \
#     --random_prune \
#     --random_prune_seed 42 123 456 \
#     --random_prune_fractions $random_prune_fractions ;


env CUDA_VISIBLE_DEVICES=$2 python3 MathNeuroFast.py \
    --model $model \
    --save_path ${save_root}/results_gsm8k_race_french \
    --train_dataset $FRENCH_GSM8K_PATH \
    --train_lm_eval_task $FRENCH_LM_EVAL_TASK \
    --eval_datasets race_fr \
    --calibration_datasets $FRENCH_RACE_CALIBRATION_DATASET \
    --calibration_dataset_names Race \
    --eval_dataset_subset 200 \
    --num_samples 500 \
    --random_state 1 \
    --scalar $scalar \
    --batch_size $batch_size \
    --proportion $proportions \
    --num_repeats 3 \
    --random_prune \
    --random_prune_seed 42 123 456 \
    --random_prune_fractions $random_prune_fractions ;