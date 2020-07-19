from cotk.metric import MetricBase

class SingleTurnDistinct(MetricBase):
    _name = 'SingleTurnDistinct'
    _version = 1

    def __init__(self, dataloader, gen_key="gen"):
        super().__init__(self._name, self._version)
        self.dataloader = dataloader
        self.gen_key = gen_key
        self.hyps = []

    def forward(self, data):
        gen = data[self.gen_key]
        for gen_sen in gen:
            self.hyps.append(self.dataloader.trim(gen_sen))

    def calc_distinct_k(self, k, gen=None):
        def h(vs, l):
            ret = 0
            for v in l:
                ret = ret * vs + v
            return ret

        src = gen
        if src is None:
            src = self.hyps
        d = {}
        vs = self.dataloader.vocab_size
        tot = 0
        for sen in src:
            for i in range(0, len(sen)-k):
                key = h(vs, sen[i:i+k])
                if key in d:
                    d[key] += 1
                else:
                    d[key] = 1
                tot += 1
        return len(d) / tot

    def close(self):
        ret = {}
        for k in range(1, 5):
            ret["distict_%d" % k] = self.calc_distinct_k(k)
        return ret

class MultiTurnDistinct(MetricBase):
    _name = 'MultiTurnDistinct'
    _version = 1

    def __init__(self, dataloader, gen_key="gen", turn_len_key="turn_length"):
        super().__init__(self._name, self._version)
        self.dataloader = dataloader
        self.turn_len_key = turn_len_key
        self.gen_key = gen_key
        self.hyps = []

    def calc_distinct_k(self, k, gen=None):
        def h(vs, l):
            ret = 0
            for v in l:
                ret = ret * vs + v
            return ret

        src = gen
        if src is None:
            src = self.hyps
        d = {}
        vs = self.dataloader.vocab_size
        tot = 0
        for sen in src:
            for i in range(0, len(sen)-k):
                key = h(vs, sen[i:i+k])
                if key not in d:
                    d[key] += 1
                else:
                    d[key] = 1
                tot += 1
        return len(d) / tot

    def forward(self, data):
        length = data[self.turn_len_key]
        gen = data[self.gen_key]
        if len(length) != len(gen):
            raise ValueError("Batch num is not matched.")

        for i, turn_length in enumerate(length):
            gen_session = gen[i]
            for j in range(turn_length):
                self.hyps.append(self.dataloader.trim(gen_session[j]))

    def close(self):
        ret = {}
        for k in range(1, 5):
            ret["distict_%d" % k] = self.calc_distinct_k(k)
        return ret