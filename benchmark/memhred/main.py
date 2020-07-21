import os

import json
import numpy as np
import tensorflow as tf
from myCoTK.dataloader import MyMemHRED
from myCoTK.wordvector import TencentChinese
from utils import debug, try_cache

from model import HredModel

def create_model(sess, data, args, embed):
	with tf.variable_scope(args.name):
		model = HredModel(data, args, embed)
		model.print_parameters()
		latest_dir = '%s/checkpoint_latest' % args.model_dir
		best_dir = '%s/checkpoint_best' % args.model_dir
		if not os.path.isdir(args.model_dir):
			os.mkdir(args.model_dir)
		if not os.path.isdir(latest_dir):
			os.mkdir(latest_dir)
		if not os.path.isdir(best_dir):
			os.mkdir(best_dir)
		if tf.train.get_checkpoint_state(latest_dir, args.name) and args.restore == "last":
			print("Reading model parameters from %s" % tf.train.latest_checkpoint(latest_dir, args.name))
			model.latest_saver.restore(sess, tf.train.latest_checkpoint(latest_dir, args.name))
		else:
			if tf.train.get_checkpoint_state(best_dir, args.name) and args.restore == "best":
				print('Reading model parameters from %s' % tf.train.latest_checkpoint(best_dir, args.name))
				model.best_saver.restore(sess, tf.train.latest_checkpoint(best_dir, args.name))
			else:
				print("Created model with fresh parameters.")
				global_variable = [gv for gv in tf.global_variables() if args.name in gv.name]
				sess.run(tf.variables_initializer(global_variable))

	return model


def main(args):
	if args.debug:
		debug()

	if args.cuda:
		if not "CUDA_VISIBLE_DEVICES" in os.environ:
			os.environ["CUDA_VISIBLE_DEVICES"] = '1'
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
	else:
		config = tf.ConfigProto(device_count={'GPU': 0})
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	np.random.seed(233)
	tf.set_random_seed(233)

	data_class = MyMemHRED
	wordvec_class = TencentChinese
	if args.cache:
		if not os.path.isdir(args.cache_dir):
			os.mkdir(args.cache_dir)
		data = try_cache(data_class, (args.datapath,), args.cache_dir)
		vocab = data.vocab_list
		embed = try_cache(lambda wv, ez, vl: wordvec_class(wv).load_matrix(ez, vl),
						  (args.wvpath, args.embedding_size, vocab),
						  args.cache_dir, wordvec_class.__name__)
	else:
		data = data_class(args.datapath)
		wv = wordvec_class(args.wvpath)
		vocab = data.vocab_list
		embed = wv.load_matrix(args.embedding_size, vocab)

	embed = np.array(embed, dtype = np.float32)
	if not os.path.isdir(args.out_dir):
		os.mkdir(args.out_dir)

	with tf.Session(config=config) as sess:
		model = create_model(sess, data, args, embed)
		if args.mode == "train":
			model.train_process(sess, data, args)
		else:
			model.test_process(sess, data, args)