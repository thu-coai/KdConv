#!/usr/bin/env bash

python run_BERTRetrieval.py \
	--do_train \
	--train_batch_size 16 \
	--learning_rate 5e-5 \
	--cache \
	--cache_dir ./cache/music \
	--datapath ../data/music \
	--num_train_epochs 3.0 \
	--output_dir ./output/music \
    --model_dir ./model/music \
    --gradient_accumulation_steps 4