MODEL="qwen2_5_vl_7b-mix-data-v2.9-huber-reward-epoch4-rl"
TASKS=("gdp" "pop" "mental" "public" "bachelor" "drive" "build_height" "violent" "acc2health" "life_exp" "house_price")
PORT=8194
DOMAIN="out"
VERSION="v2.8"

if [[ "$MODEL" == *sft* ]]; then
    MODE="sft"
else
    MODE="rl"
fi

for TASK in "${TASKS[@]}"; do
    echo "Current type: $TASK"

    python -m evaluate.eval_local --model_name=$MODEL --task_name=$TASK --domain=$DOMAIN --port=$PORT --eval_version=$VERSION --training_mode=$MODE

    python -m evaluate.metrics --model_name=$MODEL --task_name=$TASK --domain=$DOMAIN --eval_version=$VERSION  --training_mode=$MODE

done
