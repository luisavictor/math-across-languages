models=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
)

for model in "${models[@]}"; do
    bash scripts/bash_scripts/french_experiments.sh "$model"
done