#!/usr/bin/env bash

python run_BERTRetrieval.py \
	--do_train \
	--train_batch_size 16 \
	--learning_rate 5e-5 \
	--cache \
	--cache_dir ./cache/travel \
	--datapath ../data/travel \
	--num_train_epochs 8.0 \
	--output_dir ./output/travel \
    --model_dir ./model/travel \
    --gradient_accumulation_steps 8