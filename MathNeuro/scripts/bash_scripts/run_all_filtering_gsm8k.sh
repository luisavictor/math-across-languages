models=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
)

for model in "${models[@]}"; do
    bash /raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/scripts/bash_scripts/filtering_gsm8k_experiments.sh "$model"
done