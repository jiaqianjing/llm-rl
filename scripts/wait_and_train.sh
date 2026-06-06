#!/bin/bash
# 等待 GPU 4-7 持续空闲 30 分钟后自动启动 DPO 训练

PROJECT=/mnt/nvme6/ken/rl/llm-rl
IDLE_THRESHOLD=1800   # 30 分钟
CHECK_INTERVAL=60     # 每 60 秒检查一次
IDLE_FILE=/tmp/gpu_idle_start_dpo.txt

echo "[$(date '+%H:%M:%S')] 开始监控 GPU 4-7，空闲 30 分钟后启动 DPO 训练..."

while true; do
    idle=true
    for idx in 4 5 6 7; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $idx)
        [ "$mem" -gt 5000 ] && idle=false && break
    done

    now=$(date +%s)

    if $idle; then
        if [ -f "$IDLE_FILE" ]; then
            start=$(cat "$IDLE_FILE")
            elapsed=$(( now - start ))
            remaining=$(( IDLE_THRESHOLD - elapsed ))
            echo "[$(date '+%H:%M:%S')] GPU 空闲中，已等待 ${elapsed}s，还需 ${remaining}s..."

            if [ $elapsed -ge $IDLE_THRESHOLD ]; then
                echo "[$(date '+%H:%M:%S')] ✅ 空闲满 30 分钟，启动 DPO 训练！"
                rm -f "$IDLE_FILE"

                cd "$PROJECT"
                CUDA_VISIBLE_DEVICES=4,5,6,7 \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                PYTHONPATH=/mnt/nvme6/ken/opt-pytorch-pkgs:/opt/pytorch/lib/python3.12/site-packages \
                /opt/pytorch/bin/python3 -m accelerate.commands.accelerate_cli launch \
                    --config_file configs/fsdp_4gpu.yaml \
                    train_dpo.py \
                    --config configs/dpo_gsm8k.yaml \
                    2>&1 | tee /tmp/dpo_train.log

                echo "[$(date '+%H:%M:%S')] DPO 训练结束。"
                exit 0
            fi
        else
            echo "$now" > "$IDLE_FILE"
            echo "[$(date '+%H:%M:%S')] GPU 变空闲，开始计时..."
        fi
    else
        if [ -f "$IDLE_FILE" ]; then
            echo "[$(date '+%H:%M:%S')] GPU 被占用，重置计时器"
            rm -f "$IDLE_FILE"
        fi
    fi

    sleep $CHECK_INTERVAL
done
