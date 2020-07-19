#!/usr/bin/env bash
python run.py --datapath ../data/travel \
    --out_dir ./output/travel \
    --log_dir ./tensorboard/travel \
    --model_dir ./model/travel \
    --cache_dir ./cache/travel \
    --mode train \
    --cache
