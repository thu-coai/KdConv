from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import jieba
import json

class MyMetrics(object):
    def __init__(self):
        self.refs = []
        self.hyps = []

    def forword(self, ref, hyp):
        self.refs.append([jieba.lcut(ref)])
        self.hyps.append(jieba.lcut(hyp))

    def calc_distinct_k(self, k):
        d = {}
        tot = 0
        for sen in self.hyps:
            for i in range(0, len(sen)-k):
                key = ''.join(sen[i:i+k])
                d[key] = 1
                tot += 1
        return len(d) / tot

    def close(self):
        result = {}
        for i in range(1, 5):
            result["distict_%d" % i] = self.calc_distinct_k(i)
            try:
                weights = [1 / i] * i + [0] * (4 - i)
                result.update(
                    {"bleu-%d" % i: corpus_bleu(self.refs, self.hyps, weights, smoothing_function=SmoothingFunction().method3)})
            except ZeroDivisionError as _:
                result.update({"bleu-%d" % i: 0})

        return result