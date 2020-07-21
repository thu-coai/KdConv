#!/usr/bin/env bash

python run_BERTRetrieval.py \
	--do_predict \
	--predict_batch_size 16 \
	--cache \
	--num_train_epochs 8.0 \
	--cache_dir ./cache/music \
	--datapath ../data/music \
	--output_dir ./output/music \
    --model_dir ./model/music