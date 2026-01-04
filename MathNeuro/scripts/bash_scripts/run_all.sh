models=(
    "allenai/Olmo-3-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
)

for model in "${models[@]}"; do
    bash MathNeuro/scripts/bash_scripts/scaling_pruning_finetuning.sh "$model"
done