#!/usr/bin/env bash

set -o nounset
set -o pipefail
set -o errexit
set -x

epochs=50
attn_epochs=50
attn_batch_size=16
base_dir=$(readlink -m logs/paper_logs)
save_dir="$base_dir/save"
VENV="/home/$USER/local/evidence_inference_venv/bin/python"
base_cmd="$VENV evidence_inference/experiments/model_0_paper_experiment.py --epochs=$epochs --attn_epochs=$attn_epochs --attn_batch_size=$attn_batch_size --mode=paper "
logs="$base_dir/logs"
ckpts="$base_dir/ckpts"
mkdir -p $ckpts $logs $save_dir

function ckpt {
    local cmd=$1
    local name=$2
    local log=$3
    local ckpts_dir=$4

    local ckpt_file="$ckpts_dir/$name"

    if [ ! -e "$ckpts_dir/$name" ] ; then
        CUDA_VISIBLE_DEVICES=0 $cmd > >(tee ${log}.o) 2> >(tee ${log}.e >&2)
        touch "$ckpt_file"
    else
        echo "already ran $name; clear '$ckpt_file' to restart"
    fi
}

for run in `seq 1 1 1`; do
    # base paper models
    config_base="--article_sections=all --article_encoder=GRU --ico_encoder=CBoW"
    for config in \
        "$config_base"\
        "$config_base --attn"\
        "$config_base --attn --pretrain_attention=pretrain_tokenwise_attention --tokenwise"\
        "$config_base --attn --cond"\
        "$config_base --attn --cond --pretrain_attention=pretrain_tokenwise_attention --tokenwise"\
        ; do
        name=$(echo "$config" | tr ' ' ',').run-$run
        ckpt "$base_cmd $config" "$name" "$logs/$name" "$ckpts"
    done

    # oracle
    config="$config_base --data_config=cheating --attn --cond --tokenwise"
    name=$(echo "$config" | tr ' ' ',').run-$run
    ckpt "$base_cmd $config" "$name" "$logs/$name" "$ckpts"

    # no prompt
    config="$config_base --data_config=no_prompt"
    name=$(echo "$config" | tr ' ' ',').run-$run
    ckpt "$base_cmd $config" "$name" "$logs/$name" "$ckpts"

    # no article
    config="$config_base --data_config=no_article"
    name=$(echo "$config" | tr ' ' ',').run-$run
    ckpt "$base_cmd $config" "$name" "$logs/$name" "$ckpts"
done
