#!/usr/bin/env bash

python run_BERTRetrieval.py \
	--do_predict \
	--predict_batch_size 16 \
	--cache \
	--cache_dir ./cache/music \
	--num_train_epochs 3.0 \
	--datapath ../data/music \
	--output_dir ./output/music \
    --model_dir ./model/music