
model=$1

proportions="0.000001 0.00001 0.0001 0.001 0.005 0.01 0.025 0.05 0.1 0.15"
ENGLISH_MMLU_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/mmlu_val.csv"
ENGLISH_RACE_CALIBRATION_DATASET="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/race.csv"


if [[ $model == *"Qwen"* ]]; then
    export batch_size="8"
elif [[ $model == *"Llama-3.2-1B-Instruct"* ]]; then
    export batch_size="10"
else
    export batch_size="6"
fi
    

# a for loop with 2 iterations
for i in {1..2}
do
    echo "Iteration $i: Running scaling/pruning for model: $model";
    # if i == 1 --> scaler = 1.01, else scaler = 0
    if [ $i -eq 1 ]
    then
        if [[ $model == *"Llama-3.2-1B-Instruct"* ]]; then
            scalar=1.1
        else
            scalar=1.01
        fi
        save_root="/raid/s3/opengptx/behzad_shomali/LabTest/results/scale"
    else
        scalar=0
        save_root="/raid/s3/opengptx/behzad_shomali/LabTest/results/prune"
    fi


    # Coding Alpaca MMLU (scale/prune)
    CUDA_VISIBLE_DEVICES=1 python3 MathNeuroFast.py \
        --model $model \
        --save_path ${save_root}/results_codealpaca_mmlu \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/CodeAlpaca/codealpaca_train.csv \
        --train_lm_eval_task codealpaca \
        --eval_datasets mmlu \
        --calibration_datasets $ENGLISH_MMLU_CALIBRATION_DATASET \
        --calibration_dataset_names MMLU \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --proportion $proportions \
        --random_state 1 \
        --run_codealpaca_eval \
        --pre_train_eval \
        --scalar $scalar \
        --batch_size $batch_size \
        --num_repeats 3;
    
    # Coding Alpaca RACE (scale/prune)
    CUDA_VISIBLE_DEVICES=1 python3 MathNeuroFast.py \
        --model $model \
        --save_path ${save_root}/results_codealpaca_race \
        --train_dataset /raid/s3/opengptx/behzad_shomali/LabTest/custom_datasets/CodeAlpaca/codealpaca_train.csv \
        --train_lm_eval_task codealpaca \
        --eval_datasets race \
        --calibration_datasets $ENGLISH_RACE_CALIBRATION_DATASET \
        --calibration_dataset_names RACE \
        --eval_dataset_subset 200 \
        --num_samples 500 \
        --proportion $proportions \
        --random_state 1 \
        --run_codealpaca_eval \
        --pre_train_eval \
        --scalar $scalar \
        --batch_size $batch_size \
        --num_repeats 3;

    echo "Finished iteration $scalar for model: $model";
done

wait