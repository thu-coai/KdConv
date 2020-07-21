# coding:utf-8

def run(*argv):
	import argparse
	import time

	from utils import Storage

	parser = argparse.ArgumentParser(description='A hred model')
	args = Storage()

	parser.add_argument('--name', type=str, default='hred',
		help='The name of your model, used for variable scope and tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
	parser.add_argument('--restore', type=str, default='best',
		help='Checkpoints name to load. "last" for last checkpoints, "best" for best checkpoints on dev. Attention: "last" and "best" wiil cause unexpected behaviour when run 2 models in the same dir at the same time. Default: None (don\'t load anything)')
	parser.add_argument('--mode', type=str, default="train",
		help='"train" or "test". Default: train')
	parser.add_argument('--dataset', type=str, default='MyMemHRED',
		help='Dataloader class. Default: UbuntuCorpus')
	parser.add_argument('--datapath', type=str, default='../data/film',
		help='Directory for data set. Default: UbuntuCorpus')
	parser.add_argument('--epoch', type=int, default=20,
		help="Epoch for trainning. Default: 100")
	parser.add_argument('--wvclass', type=str, default='TencentChinese',
		help="Wordvector class, none for not using pretrained wordvec. Default: Glove")
	parser.add_argument('--wvpath', type=str, default="wordvector/chinese",
		help="Directory for pretrained wordvector. Default: resources://Glove300d")

	parser.add_argument('--out_dir', type=str, default="./output/film",
		help='Output directory for test output. Default: ./output')
	parser.add_argument('--log_dir', type=str, default="./tensorboard/film",
		help='Log directory for tensorboard. Default: ./tensorboard')
	parser.add_argument('--model_dir', type=str, default="./model/film",
		help='Checkpoints directory for model. Default: ./model')
	parser.add_argument('--cache_dir', type=str, default="./cache/film",
		help='Checkpoints directory for cache. Default: ./cache')
	parser.add_argument('--cpu', action="store_true",
		help='Use cpu.')
	parser.add_argument('--debug', action='store_true',
		help='Enter debug mode (using ptvsd).')
	parser.add_argument('--cache', action='store_true',
		help='Use cache for speeding up load data and wordvec. (It may cause problems when you switch dataset.)')
	cargs = parser.parse_args(argv)

	# Editing following arguments to bypass command line.
	args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())
	args.restore = cargs.restore
	args.mode = cargs.mode
	args.dataset = cargs.dataset
	args.datapath = cargs.datapath
	args.epochs = cargs.epoch
	args.wvclass = cargs.wvclass
	args.wvpath = cargs.wvpath
	args.out_dir = cargs.out_dir
	args.log_dir = cargs.log_dir
	args.model_dir = cargs.model_dir
	args.cache_dir = cargs.cache_dir
	args.debug = cargs.debug
	args.cache = cargs.cache
	args.cuda = not cargs.cpu

	args.softmax_samples = 512
	args.embedding_size = 200
	args.eh_size = 200
	args.ch_size = 200
	args.dh_size = 200
	args.lr = 1e-3
	args.lr_decay = 0.99
	args.batch_size = 32
	args.grad_clip = 5.0
	args.show_sample = [0]
	args.max_sent_length = 50
	args.checkpoint_steps = 100
	args.checkpoint_max_to_keep = 5

	import random
	random.seed(0)

	from main import main

	main(args)

if __name__ == '__main__':
	import sys
	run(*sys.argv[1:])
