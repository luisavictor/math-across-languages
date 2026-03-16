
proportions="0.000001 0.00001 0.0001 0.001 0.01 0.1"

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

model=$1
if [[ $model == *"Qwen"* ]]; then
    export batch_size="18"
    export hindi_correct_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_hindi/train_correct_Qwen_Qwen3-4B-Instruct-2507.csv"
    export hindi_incorrect_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_hindi/train_incorrect_Qwen_Qwen3-4B-Instruct-2507.csv"
    export german_correct_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_german/train_correct_Qwen_Qwen3-4B-Instruct-2507.csv"
    export german_incorrect_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_german/train_incorrect_Qwen_Qwen3-4B-Instruct-2507.csv"
    export english_correct_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k/train_correct_Qwen_Qwen3-4B-Instruct-2507.csv"
    export english_incorrect_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k/train_incorrect_Qwen_Qwen3-4B-Instruct-2507.csv"
elif [[ $model == *"Llama-3.2-1B-Instruct"* ]]; then
    export batch_size="20"
    export hindi_correct_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_hindi/train_correct_meta-llama_Llama-3.2-1B-Instruct.csv"
    export hindi_incorrect_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_hindi/train_incorrect_meta-llama_Llama-3.2-1B-Instruct.csv"
    export german_correct_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_german/train_correct_meta-llama_Llama-3.2-1B-Instruct.csv"
    export german_incorrect_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_german/train_incorrect_meta-llama_Llama-3.2-1B-Instruct.csv"
    export english_correct_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k/train_correct_meta-llama_Llama-3.2-1B-Instruct.csv"
    export english_incorrect_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k/train_incorrect_meta-llama_Llama-3.2-1B-Instruct.csv"
else
    export batch_size="16"
    export hindi_correct_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_hindi/train_correct_meta-llama_Llama-3.1-8B-Instruct.csv"
    export hindi_incorrect_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_hindi/train_incorrect_meta-llama_Llama-3.1-8B-Instruct.csv"
    export german_correct_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_german/train_correct_meta-llama_Llama-3.1-8B-Instruct.csv"
    export german_incorrect_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k_german/train_incorrect_meta-llama_Llama-3.1-8B-Instruct.csv"
    export english_correct_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k/train_correct_meta-llama_Llama-3.1-8B-Instruct.csv"
    export english_incorrect_train_dataset="/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/data/filtered_gsm8k/train_incorrect_meta-llama_Llama-3.1-8B-Instruct.csv"
fi


# a for loop with 2 iterations
for i in {1..2}
do
    for j in {1..2}
    do
        if [ $i -eq 1 ]
        then
            scalar=1.01
            save_root="/raid/s3/opengptx/behzad_shomali/LabTest/results_correct_vs_incorrect/scale"
        else
            scalar=0
            save_root="/raid/s3/opengptx/behzad_shomali/LabTest/results_correct_vs_incorrect/prune"
        fi

        if [ $j -eq 1 ]
        then
            data_mode="correct_only"
            save_root="${save_root}/correct_only"
        else
            data_mode="false_only"
            save_root="${save_root}/false_only"
        fi
        echo "Iteration $i: Running scaling/pruning for model: $model with data mode: $data_mode";

        ## Hindi RACE 
        env CUDA_VISIBLE_DEVICES=1 python3 MathNeuro_SelectedSamples.py \
            --model $model \
            --save_path ${save_root}/results_gsm8k_race_hindi_max300 \
            --train_dataset $HINDI_GSM8K_PATH \
            --train_lm_eval_task $HINDI_LM_EVAL_TASK \
            --eval_datasets race_hi_max300 \
            --calibration_datasets $HINDI_RACE_CALIBRATION_DATASET \
            --calibration_dataset_names Race \
            --eval_dataset_subset 200 \
            --num_samples 500 \
            --random_state 1 \
            --scalar $scalar \
            --batch_size $batch_size \
            --proportion $proportions \
            --num_repeats 3 \
            --text_file "parameter_statistics" \
            --data_mode $data_mode \
            --correct_train_dataset $hindi_correct_train_dataset \
            --false_train_dataset $hindi_incorrect_train_dataset &
        sleep 20s;

        ## German RACE 
        env CUDA_VISIBLE_DEVICES=2 python3 MathNeuro_SelectedSamples.py \
            --model $model \
            --save_path ${save_root}/results_gsm8k_race_german \
            --train_dataset $GERMAN_GSM8K_PATH \
            --train_lm_eval_task $GERMAN_LM_EVAL_TASK \
            --eval_datasets race_de \
            --calibration_datasets $GERMAN_RACE_CALIBRATION_DATASET \
            --calibration_dataset_names Race \
            --eval_dataset_subset 200 \
            --num_samples 500 \
            --random_state 1 \
            --scalar $scalar \
            --batch_size $batch_size \
            --proportion $proportions \
            --num_repeats 3 \
            --text_file "parameter_statistics" \
            --data_mode $data_mode \
            --correct_train_dataset $german_correct_train_dataset \
            --false_train_dataset $german_incorrect_train_dataset &
        sleep 20s;

        ## English RACE 
        env CUDA_VISIBLE_DEVICES=3 python3 MathNeuro_SelectedSamples.py \
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
            --num_repeats 3 \
            --text_file "parameter_statistics" \
            --data_mode $data_mode \
            --correct_train_dataset $english_correct_train_dataset \
            --false_train_dataset $english_incorrect_train_dataset &
        sleep 5s;
        wait
    done
done

wait