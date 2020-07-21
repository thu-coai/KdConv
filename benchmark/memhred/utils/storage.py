# coding:utf-8

class Storage(dict):
	def __init__(self, *args, **kw):
		dict.__init__(self, *args, **kw)

	def __getattr__(self, key):
		if key in self:
			return self[key]
		else:
			return getattr(super(Storage, self), key)

	def __setattr__(self, key, value):
		self[key] = value

	def __delattr__(self, key):
		del self[key]

	def __sub__(self, b):
		'''Delete all items which b has (ignore values).
		'''
		res = Storage()
		for i, j in self.items():
			if i not in b:
				res[i] = j
			elif isinstance(j, Storage) and isinstance(b[i], Storage):
				diff = j - b[i]
				res[i] = diff
		return res

	def __xor__(self, b):
		'''Return different items in two Storages (only intersection keys).
		'''
		res = Storage()
		for i, j in self.items():
			if i in b:
				if isinstance(j, Storage) and isinstance(b[i], Storage):
					res[i] = j ^ b[i]
				elif j != b[i]:
					res[i] = (j, b[i])
		return res

	def update(self, b):
		'''will NOT overwrite existed key.
		'''
		for i, j in b.items():
			if i not in self:
				self[i] = j
			elif isinstance(self[i], Storage) and isinstance(j, Storage):
				self[i].update(j)
