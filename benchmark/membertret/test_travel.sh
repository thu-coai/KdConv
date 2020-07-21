#!/usr/bin/env bash

python run_BERTRetrieval.py \
	--do_predict \
	--predict_batch_size 16 \
	--cache \
	--num_train_epochs 8.0 \
	--cache_dir ./cache/travel \
	--datapath ../data/travel \
	--output_dir ./output/travel \
    --model_dir ./model/travel