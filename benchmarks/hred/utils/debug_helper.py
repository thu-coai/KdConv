import logging

def debug():
	import ptvsd
	ptvsd.enable_attach()
	logging.info("wait debug")
	ptvsd.wait_for_attach()
