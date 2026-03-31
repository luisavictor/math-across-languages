
proportions="0.001 0.01 0.1"
# random_prune_fractions="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
random_prune_fractions="0.05"

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

model="meta-llama/Llama-3.2-1B-Instruct"
if [[ $model == *"Qwen"* ]]; then
    export batch_size="18"
elif [[ $model == *"Llama-3.2-1B-Instruct"* ]]; then
    export batch_size="20"
else
    export batch_size="16"
fi


echo "Running scaling/pruning for model: $model";

scalar=0
save_root="/raid/s3/opengptx/behzad_shomali/LabTest/results/prune"

## English RACE 
CUDA_VISIBLE_DEVICES=4 python3 MathNeuroFast.py \
    --model $model \
    --save_path ${save_root}/results_gsm8k_race \
    --train_dataset $ENGLISH_GSM8K_PATH \
    --train_lm_eval_task $ENGLISH_LM_EVAL_TASK \
    --eval_datasets race \
    --calibration_datasets $ENGLISH_RACE_CALIBRATION_DATASET \
    --calibration_dataset_names Race \
    --eval_dataset_subset 200 \
    --num_samples 500 \
    --random_state 1 \
    --scalar $scalar \
    --batch_size $batch_size \
    --proportion $proportions \
    --num_repeats 1 \
    --random_prune \
    --random_prune_seed 42 \
    --random_prune_fractions $random_prune_fractions;

sleep 5s;
