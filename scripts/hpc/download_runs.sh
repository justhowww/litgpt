#!/bin/bash

REMOTE_HOST="huangyh@login.zaratan.umd.edu"
REMOTE_ROOT="/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data/runs"
LOCAL_ROOT="/Users/justinhuang/Documents/lab/Chris/2025spring-diffusion-video-decoder/results"

dir_name="$1"

if [ -z "$dir_name" ]; then
    echo "Usage: $0 <run_dir_name>"
    exit 1
fi

remote_dir="$REMOTE_ROOT/$dir_name"
remote_tb_dir="$remote_dir/logs/tensorboard"
local_tb_dir="$LOCAL_ROOT/$dir_name/tensorboard"



echo "[SYNC] $REMOTE_HOST:$remote_tb_dir -> $local_tb_dir"
mkdir -p "$local_tb_dir"

rsync -avz --progress \
    "$REMOTE_HOST:$remote_tb_dir/" \
    "$local_tb_dir/"