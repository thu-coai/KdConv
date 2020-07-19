import pickle
import os
import json

def try_cache(module, args, cache_dir, name=None):
	'''Cache a function return value in cache.
	If already existed, read cache.
	'''
	if name is None:
		name = module.__name__
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)
	fname = "%s/%s.pkl" % (cache_dir, name)
	if os.path.exists(fname):
		f = open(fname, "rb")
		info, obj = pickle.load(f)
		f.close()
	else:
		info = None
		obj = None
	if info != args:
		obj = module(*args)
		f = open(fname, "wb")
		pickle.dump((args, obj), f, -1)
		f.close()
	return obj
