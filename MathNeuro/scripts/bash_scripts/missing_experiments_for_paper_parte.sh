
# proportions="0.000001 0.00001 0.0001 0.001 0.005 0.01 0.025 0.05 0.1 0.15"
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


# Table 1

## Qwen Hindi RACE (scale)
CUDA_VISIBLE_DEVICES=2 python3 MathNeuroFast.py \
    --model "Qwen/Qwen3-4B-Instruct-2507" \
    --save_path /raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_hindi_max300 \
    --train_dataset $HINDI_GSM8K_PATH \
    --train_lm_eval_task $HINDI_LM_EVAL_TASK \
    --eval_datasets race_hi_max300 \
    --calibration_datasets $HINDI_RACE_CALIBRATION_DATASET \
    --calibration_dataset_names Race \
    --eval_dataset_subset 200 \
    --num_samples 500 \
    --random_state 1 \
    --scalar 1.01 \
    --batch_size "5" \
    --proportion 0.005 0.025 \
    --num_repeats 3;


sleep 5s;

wait