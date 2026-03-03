ENGLISH_TASK="gsm8k_cot_train"
ENGLISH_GSM8K_PATH="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/gsm8k.csv"

HINDI_TASK="gsm8k_hi_cot_train"
HINDI_GSM8K_PATH="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_Hindi/gsm8k_hi_train.csv"

GERMAN_TASK="gsm8k_de_cot_train"
GERMAN_GSM8K_PATH="/raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/GSM8k_German/gsm8k_de_train.csv"


model=$1
if [[ $model == *"Qwen"* ]]; then
    export batch_size="14"
elif [[ $model == *"Llama-3.2-1B-Instruct"* ]]; then
    export batch_size="18"
else
    export batch_size="12"
fi

env CUDA_VISIBLE_DEVICES=3 python3 /raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/filter_training_data.py \
    --model $model \
    --train_data $ENGLISH_GSM8K_PATH \
    --task_name $ENGLISH_TASK \
    --output_dir "/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k" \
    --samples_per_group 2500 \
    --batch_size $batch_size \
    --random_seed 1 &

sleep 20s;

env CUDA_VISIBLE_DEVICES=5 python3 /raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/filter_training_data.py \
    --model $model \
    --train_data $HINDI_GSM8K_PATH \
    --task_name $HINDI_TASK \
    --output_dir "/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_hindi" \
    --samples_per_group 2500 \
    --batch_size $batch_size \
    --random_seed 1 &

sleep 20s;

env CUDA_VISIBLE_DEVICES=1 python3 /raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/filter_training_data.py \
    --model $model \
    --train_data $GERMAN_GSM8K_PATH \
    --task_name $GERMAN_TASK \
    --output_dir "/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_german" \
    --samples_per_group 2500 \
    --batch_size $batch_size \
    --random_seed 1;