#!/usr/bin/env bash

python run_BERTRetrieval.py \
	--do_train \
	--train_batch_size 16 \
	--learning_rate 5e-5 \
	--cache \
	--cache_dir ./cache/film \
	--datapath ../data/film \
	--num_train_epochs 8.0 \
	--output_dir ./output/film \
    --model_dir ./model/film \
    --gradient_accumulation_steps 8