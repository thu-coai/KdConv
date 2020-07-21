import os
import time
from collections import Counter
from itertools import chain
import multiprocessing
from multiprocessing import Pool
import tqdm

import jieba
import json

import numpy as np

from cotk._utils.file_utils import get_resource_file_path
from cotk.dataloader import LanguageProcessingBase, BERTLanguageProcessingBase, SingleTurnDialog
from cotk.metric import MetricChain, PerplexityMetric, SingleTurnDialogRecorder
from ..metric import SingleTurnResponseRecorder, BleuCorpusMetric, SingleTurnDistinct


class MyLM(SingleTurnDialog):
    def __init__(self, file_id="../data/film", min_vocab_times=0,
                 max_sent_length=10086, invalid_vocab_times=0):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        super(MyLM, self).__init__()


    def _load_data(self):
        r'''Loading dataset, invoked during the initialization of :class:`SingleTurnDialog`.
        '''
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        for key in self.key_name:
            origin_data[key] = {'post': [], 'resp': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key)))

            for data in datas:
                messages = data['messages']
                i = 0
                if key != 'test':
                    resp_sent = []
                    while i + 1 < len(messages):

                        if i == 0:
                            tmp_sent = jieba.lcut(messages[i]['message'])
                            resp_sent = tmp_sent

                            count_token(tmp_sent)
                            if 'attrs' in messages[0]:
                                for attr in messages[0]['attrs']:
                                    h = jieba.lcut(attr['zsname'])
                                    r = jieba.lcut(attr['zsattrname'])
                                    t = jieba.lcut(attr['zsattrvalue'])
                                    count_token(h + r + t)

                        nxt_sent = jieba.lcut(messages[i + 1]['message'])
                        resp_sent = resp_sent + ['<eos>'] + nxt_sent

                        count_token(nxt_sent)
                        if 'attrs' in messages[i + 1]:
                            for attr in messages[i + 1]['attrs']:
                                h = jieba.lcut(attr['zsname'])
                                r = jieba.lcut(attr['zsattrname'])
                                t = jieba.lcut(attr['zsattrvalue'])
                                count_token(h + r + t)

                        i += 1


                    origin_data[key]['post'].append([])
                    origin_data[key]['resp'].append(resp_sent)

                else:
                    post_sent = []
                    while i + 1 < len(messages):

                        if i == 0:
                            post_sent = jieba.lcut(messages[0]['message'])
                            count_token(post_sent)
                            if 'attrs' in messages[0]:
                                for attr in messages[0]['attrs']:
                                    h = jieba.lcut(attr['zsname'])
                                    r = jieba.lcut(attr['zsattrname'])
                                    t = jieba.lcut(attr['zsattrvalue'])
                                    count_token(h + r + t)

                        nxt_sent = jieba.lcut(messages[i + 1]['message'])
                        origin_data[key]['post'].append(post_sent)
                        origin_data[key]['resp'].append(nxt_sent)

                        post_sent = post_sent + ['<eos>'] + nxt_sent

                        count_token(nxt_sent)
                        if 'attrs' in messages[i + 1]:
                            for attr in messages[i + 1]['attrs']:
                                h = jieba.lcut(attr['zsname'])
                                r = jieba.lcut(attr['zsattrname'])
                                t = jieba.lcut(attr['zsattrvalue'])
                                count_token(h + r + t)

                        i += 1

        # Important: Sort the words preventing the index changes between different runs
        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list = self.ext_vocab + left_vocab
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        left_vocab = list(filter(lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
        vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        word2id = {w: i for i, w in enumerate(vocab_list)}
        line2id = lambda line: ([self.go_id] +
                                  list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) +
                                  [self.eos_id])[:self._max_sent_length]
        line2id_post = lambda line: ([self.go_id] +
                                  list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line))
                                  )[:self._max_sent_length]
        line2id_resp = lambda line: ([self.eos_id] +
                                  list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) +
                                  [self.eos_id])[:self._max_sent_length]

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}
            if key != 'test':
                data[key]['post'] = list(map(line2id, origin_data[key]['post']))
                data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
            else:
                data[key]['post'] = list(map(line2id_post, origin_data[key]['post']))
                data[key]['resp'] = list(map(line2id_resp, origin_data[key]['resp']))
            data_size[key] = len(data[key]['post'])

            vocab = list(chain(*(origin_data[key]['post'] + origin_data[key]['resp'])))
            vocab_num = len(vocab)
            oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
            invalid_num = len(list(filter(lambda word: word not in valid_vocab_set, vocab))) - oov_num
            length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
            cut_num = np.sum(np.maximum(np.array(length) - self._max_sent_length + 1, 0))
            print("%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, cut word rate: %f" % \
                  (key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))

        return vocab_list, valid_vocab_len, data, data_size


    def get_inference_metric(self, gen_key="gen"):
        '''Get metrics for inference.

        It contains:

        * :class:`.metric.BleuCorpusMetric`
        * :class:`.metric.SingleTurnDialogRecorder`

        Arguments:
            gen_key (str): The key of generated sentences in index form.
                Refer to :class:`.metric.BleuCorpusMetric` or
                :class:`.metric.SingleTurnDialogRecorder`. Default: ``gen``.

        Returns:
            A :class:`.metric.MetricChain` object.
        '''
        metric = MetricChain()
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric


class MySeq2Seq(SingleTurnDialog):
    def __init__(self, file_id="../data/film", min_vocab_times=0, num_turns=8,
            max_sent_length=10086, invalid_vocab_times=0):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        self._num_turns = num_turns
        super(MySeq2Seq, self).__init__()

    def _load_data(self):
        r'''Loading dataset, invoked during the initialization of :class:`SingleTurnDialog`.
        '''
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        for key in self.key_name:
            origin_data[key] = {'post': [], 'resp': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key)))

            for data in datas:
                history = []
                messages = data['messages']
                i = 0
                nxt_sent = jieba.lcut(messages[0]['message'])
                while i + 1 < len(messages):
                    tmp_sent, nxt_sent = nxt_sent, jieba.lcut(messages[i + 1]['message'])
                    history.append(tmp_sent)
                    post = []
                    for jj in range(max(-self._num_turns+1, -i-1), 0):
                        post = post + history[jj] + ['<eos>', '<go>']
                    post = post[:-2]
                    origin_data[key]['post'].append(post)
                    origin_data[key]['resp'].append(nxt_sent)

                    count_token(nxt_sent)
                    if 'attrs' in messages[i + 1]:
                        for attr in messages[i + 1]['attrs']:
                            h = jieba.lcut(attr['zsname'])
                            r = jieba.lcut(attr['zsattrname'])
                            t = jieba.lcut(attr['zsattrvalue'])
                            count_token(h + r + t)

                    if i == 0:
                        count_token(tmp_sent)
                        if 'attrs' in messages[0]:
                            for attr in messages[0]['attrs']:
                                h = jieba.lcut(attr['zsname'])
                                r = jieba.lcut(attr['zsattrname'])
                                t = jieba.lcut(attr['zsattrvalue'])
                                count_token(h + r + t)

                    i += 1

        # Important: Sort the words preventing the index changes between different runs
        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list = self.ext_vocab + left_vocab
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        left_vocab = list(filter(lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
        vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        word2id = {w: i for i, w in enumerate(vocab_list)}
        line2id = lambda line: ([self.go_id] +
                    list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) +
                    [self.eos_id])[:self._max_sent_length]

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}

            data[key]['post'] = list(map(line2id, origin_data[key]['post']))
            data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
            data_size[key] = len(data[key]['post'])

            vocab = list(chain(*(origin_data[key]['post'] + origin_data[key]['resp'])))
            vocab_num = len(vocab)
            oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
            invalid_num = len(list(filter(lambda word: word not in valid_vocab_set, vocab))) - oov_num
            length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
            cut_num = np.sum(np.maximum(np.array(length) - self._max_sent_length + 1, 0))
            print("%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, cut word rate: %f" %
                  (key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))

        return vocab_list, valid_vocab_len, data, data_size

    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {}
        batch_size = len(indexes)

        res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), indexes)))
        res["prev_length"] = np.array(list(map(lambda i: (
            len(self.data[key]['post'][i]) - self.data[key]['post'][i][::-1].index(self.eos_id, 1)
            if self.eos_id in self.data[key]['post'][i][:-1] else 0), indexes)))
        res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), indexes)))
        res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
        res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            post = self.data[key]['post'][idx]
            resp = self.data[key]['resp'][idx]
            res_post[i, :len(post)] = post
            res_resp[i, :len(resp)] = resp

        res["post_allvocabs"] = res_post.copy()
        res["resp_allvocabs"] = res_resp.copy()
        res_post[res_post >= self.valid_vocab_len] = self.unk_id
        res_resp[res_resp >= self.valid_vocab_len] = self.unk_id

        return res

    def get_inference_metric(self, gen_key="gen"):
        '''Get metric for inference.

        It contains:

        * :class:`.metric.BleuCorpusMetric`
        * :class:`.metric.MultiTurnDialogRecorder`

        Arguments:
            gen_key (str): The key of generated sentences in index form.
                Refer to :class:`.metric.BleuCorpusMetric` or :class:`.metric.MultiTurnDialogRecorder`.
                Default: ``gen``.

        Returns:
            A :class:`.metric.MetricChain` object.
        '''
        metric = MetricChain()
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnResponseRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric


class MyMemSeq2Seq(SingleTurnDialog):
    def __init__(self, file_id="../data/film", min_vocab_times=0, max_sent_length=10086, invalid_vocab_times=0, num_turns=8,
                 max_know_length=100):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        self._num_turns = num_turns
        self._max_know_length = max_know_length
        super(MyMemSeq2Seq, self).__init__()

    def _load_data(self):
        r'''Loading dataset, invoked during the initialization of :class:`SingleTurnDialog`.
        '''
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        for key in self.key_name:
            origin_data[key] = {'post': [], 'resp': [], 'kg': [], 'kg_index': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key), encoding='utf8'))

            for data in datas:
                messages = data['messages']
                kg = []
                kg_index = []
                kg_dict = {}
                for message in messages:
                    kg_index.append([])
                    if 'attrs' in message:
                        for attr in message['attrs']:
                            h = jieba.lcut(attr['zsname'])
                            r = jieba.lcut(attr['zsattrname'])
                            t = jieba.lcut(attr['zsattrvalue'])
                            k = tuple((tuple(h), tuple(r), tuple(t)))
                            if k not in kg_dict:
                                kg_dict[k] = len(kg)
                                kg.append(k)
                            kg_index[-1].append(kg_dict[k])
                            count_token(h + r + t)

                history = []
                i = 0
                nxt_sent = jieba.lcut(messages[0]['message'])
                while i + 1 < len(messages):
                    tmp_sent, nxt_sent = nxt_sent, jieba.lcut(messages[i + 1]['message'])
                    history.append(tmp_sent)
                    post = []
                    for jj in range(max(-self._num_turns + 1, -i - 1), 0):
                        post = post + history[jj] + ['<eos>', '<go>']
                    post = post[:-2]

                    count_token(nxt_sent)
                    if i == 0:
                        count_token(tmp_sent)

                    origin_data[key]['post'].append(post)
                    origin_data[key]['resp'].append(nxt_sent)
                    origin_data[key]['kg'].append(kg)
                    origin_data[key]['kg_index'].append(kg_index[i + 1])

                    i += 1

        # Important: Sort the words preventing the index changes between different runs
        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list = self.ext_vocab + left_vocab
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        left_vocab = list(filter(lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
        vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        word2id = {w: i for i, w in enumerate(vocab_list)}
        know2id = lambda line: list(map(lambda word: word2id.get(word, self.unk_id), line))
        line2id = lambda line: ([self.go_id] + list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) + [
            self.eos_id])[:self._max_sent_length]
        knows2id = lambda lines: list(map(know2id, lines))

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}
            data[key]['post'] = list(map(line2id, origin_data[key]['post']))
            data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
            data[key]['kg'] = [list(map(knows2id, kg)) for kg in origin_data[key]['kg']]
            data[key]['kg_index'] = origin_data[key]['kg_index']
            data_size[key] = len(data[key]['post'])

        return vocab_list, valid_vocab_len, data, data_size


    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {}
        batch_size = len(indexes)

        res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), indexes)))
        res["prev_length"] = np.array(list(map(lambda i: (
            len(self.data[key]['post'][i]) - self.data[key]['post'][i][::-1].index(self.eos_id, 1)
            if self.eos_id in self.data[key]['post'][i][:-1] else 0), indexes)))
        res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), indexes)))
        res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
        res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            post = self.data[key]['post'][idx]
            resp = self.data[key]['resp'][idx]
            res_post[i, :len(post)] = post
            res_resp[i, :len(resp)] = resp

        res["post_allvocabs"] = res_post.copy()
        res["resp_allvocabs"] = res_resp.copy()
        res_post[res_post >= self.valid_vocab_len] = self.unk_id
        res_resp[res_resp >= self.valid_vocab_len] = self.unk_id

        max_kg_num = max([len(self.data[key]['kg'][idx]) for idx in indexes])
        res["kg_h_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res["kg_hr_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res['kg_hrt_length'] = np.zeros((batch_size, max_kg_num), dtype=int)
        for i, idx in enumerate(indexes):
            kg_h_length = [min(self._max_know_length, len(sent[0])) for sent in self.data[key]['kg'][idx]]
            res["kg_h_length"][i, :len(kg_h_length)] = kg_h_length
            kg_hr_length = [min(self._max_know_length, len(sent[0]) + len(sent[1])) for sent in self.data[key]['kg'][idx]]
            res["kg_hr_length"][i, :len(kg_hr_length)] = kg_hr_length
            kg_hrt_length = [min(self._max_know_length, len(sent[0]) + len(sent[1]) + len(sent[2])) for sent in
                             self.data[key]['kg'][idx]]
            res["kg_hrt_length"][i, :len(kg_hrt_length)] = kg_hrt_length

        res['kg'] = np.zeros((batch_size, max_kg_num, np.max(res["kg_hrt_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            for j, tri in enumerate(self.data[key]['kg'][idx]):
                sent = (tri[0] + tri[1] + tri[2])[:self._max_know_length]
                res['kg'][i, j, :len(sent)] = sent

        res['kg_index'] = np.zeros((batch_size, max_kg_num), dtype=float)
        for i, idx in enumerate(indexes):
            for kgid in self.data[key]['kg_index'][idx]:
                res['kg_index'][i, kgid] = 1

        return res

    def get_inference_metric(self, gen_key="gen"):
        '''Get metrics for inference.

        It contains:

        * :class:`.metric.BleuCorpusMetric`
        * :class:`.metric.SingleTurnDialogRecorder`

        Arguments:
            gen_key (str): The key of generated sentences in index form.
                Refer to :class:`.metric.BleuCorpusMetric` or
                :class:`.metric.SingleTurnDialogRecorder`. Default: ``gen``.

        Returns:
            A :class:`.metric.MetricChain` object.
        '''
        metric = MetricChain()
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnResponseRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric