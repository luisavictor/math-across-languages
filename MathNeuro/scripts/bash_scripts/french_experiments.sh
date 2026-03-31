
proportions="0.000001 0.00001 0.0001 0.001 0.005 0.01 0.025 0.05 0.1 0.15"
proportions="0.01"

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

# model=$1
model="meta-llama/Llama-3.2-1B-Instruct"
if [[ $model == *"Qwen"* ]]; then
    export batch_size="16"
elif [[ $model == *"Llama-3.2-1B-Instruct"* ]]; then
    # export batch_size="18"
    export batch_size="10"
else
    export batch_size="14"
fi


# a for loop with 2 iterations
# for i in {1..2}
for i in {1..1}
do
    # if i == 1 --> scaler = 1.01, else scaler = 0
    if [ $i -eq 1 ]
    then
        device=7
        scalar=1.01
        save_root="/raid/s3/opengptx/behzad_shomali/LabTest/results/scale"
    else
        device=7
        scalar=0
        save_root="/raid/s3/opengptx/behzad_shomali/LabTest/results/prune"
    fi

    echo "Iteration $i: Running scaling/pruning for model: $model, device: $device, scalar: $scalar";

    # ## French RACE 
    env CUDA_VISIBLE_DEVICES=$device python3 MathNeuroFast.py \
        --model $model \
        --save_path ${save_root}/results_gsm8k_race_french_post_pre \
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
        --pre_train_eval \
        --num_repeats 1 ;
    sleep 20s;


    ## French MMLU
    # CUDA_VISIBLE_DEVICES=0 python3 MathNeuroFast.py \
    #     --model $model \
    #     --save_path ${save_root}/results_gsm8k_mmlu_french \
    #     --train_dataset $FRENCH_GSM8K_PATH \
    #     --train_lm_eval_task $FRENCH_LM_EVAL_TASK \
    #     --eval_datasets mmlu_fr \
    #     --calibration_datasets $FRENCH_MMLU_CALIBRATION_DATASET \
    #     --calibration_dataset_names MMLU \
    #     --eval_dataset_subset 200 \
    #     --num_samples 500 \
    #     --random_state 1 \
    #     --scalar $scalar \
    #     --batch_size $batch_size \
    #     --proportion $proportions \
    #     --pre_train_eval \
    #     --num_repeats 3;
    # sleep 5s;

done

wait