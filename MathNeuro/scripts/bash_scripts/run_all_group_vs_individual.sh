models=(
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
)

i=0
for model in "${models[@]}"; do
    bash scripts/bash_scripts/group_vs_individual_experiments.sh "$model" 5;
    # i=$((i+1))
done
wait