#!/usr/bin/env bash

python run_BERTRetrieval.py \
	--do_predict \
	--predict_batch_size 16 \
	--cache \
	--cache_dir ./cache/film \
	--num_train_epochs 3.0 \
	--datapath ../data/film \
	--output_dir ./output/film \
    --model_dir ./model/film