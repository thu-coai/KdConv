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
from cotk.dataloader import LanguageProcessingBase, BERTLanguageProcessingBase, MultiTurnDialog
from cotk.metric import MetricChain, PerplexityMetric, SingleTurnDialogRecorder
from ..metric import SingleTurnResponseRecorder, BleuCorpusMetric, SingleTurnDistinct


class MyHRED(MultiTurnDialog):
    def __init__(self, file_id="../data/film", min_vocab_times=0,
            max_sent_length=10086, invalid_vocab_times=0, num_turns=8):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        self._num_turns = num_turns
        super(MyHRED, self).__init__()

    def _load_data(self):
        r'''Loading dataset, invoked during the initialization of :class:`MultiTurnDialog`.
        '''
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        for key in self.key_name:
            origin_data[key] = {'posts': [], 'prev_posts': [], 'responses': []}
            datas = json.load(open('%s/%s.json' % (self._file_path, key)))
            for data in datas:
                messages = data['messages']
                turn = []
                for message in messages:
                    sent = jieba.lcut(message['message'])
                    turn.append(sent)

                    count_token(sent)
                    if 'attrs' in message:
                        for attr in message['attrs']:
                            h = jieba.lcut(attr['zsname'].replace('【', '').replace('】', ''))
                            r = jieba.lcut(attr['zsattrname'].replace('【', '').replace('】', ''))
                            t = jieba.lcut(attr['zsattrvalue'].replace('【', '').replace('】', ''))
                            count_token(h + r + t)

                for i in range(len(turn) - 1):
                    posts = [turn[j] for j in range(max(0, (i + 1) - (self._num_turns - 1)), i + 1)]
                    prev_post = posts[-1]
                    response = turn[i + 1]

                    origin_data[key]['posts'].append(posts)
                    origin_data[key]['prev_posts'].append(prev_post)
                    origin_data[key]['responses'].append(response)


        # Important: Sort the words preventing the index changes between different runs
        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list = self.ext_vocab + left_vocab
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        left_vocab = list(filter(
                lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
        vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        word2id = {w: i for i, w in enumerate(vocab_list)}
        line2id = lambda line: ([self.go_id] + list(map(lambda word: word2id.get(word, self.unk_id), line)) + \
                                [self.eos_id])[:self._max_sent_length]
        lines2id = lambda lines: list(map(line2id, lines))

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}
            data[key]['posts'] = list(map(lines2id, origin_data[key]['posts']))
            data[key]['responses'] = list(map(line2id, origin_data[key]['responses']))
            data[key]['prev_posts'] = list(map(line2id, origin_data[key]['prev_posts']))

            data_size[key] = len(data[key]['responses'])

        return vocab_list, valid_vocab_len, data, data_size

    def get_batch(self, key, indexes):
        '''{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}

        Returns:
            (dict): A dict at least contains:
            {MultiTurnDialog.GET_BATCH_RETURNS_DICT}

            See the example belows.

        Examples:
            {MultiTurnDialog.GET_BATCH_EXAMPLES_PART}
            }
        '''
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        batch_size = len(indexes)
        res = {}

        res['context_length'] = [len(self.data[key]['posts'][idx]) for idx in indexes]
        max_context_length = max(res['context_length'])
        res['posts_length'] = np.zeros((batch_size * max_context_length), dtype=int)
        for i, idx in enumerate(indexes):
            posts_length = [len(each) for each in self.data[key]['posts'][idx]]
            res['posts_length'][i * max_context_length: i * max_context_length + len(posts_length)] = posts_length

        max_posts_length = np.max(res['posts_length'])
        res['posts'] = np.zeros((batch_size * max_context_length, max_posts_length), dtype=int)
        for i, idx in enumerate(indexes):
            for j in range(res['context_length'][i]):
                post = self.data[key]['posts'][idx][j]
                res['posts'][i * max_context_length + j, :len(post)] = post

        res['prev_posts_length'] = [len(self.data[key]['posts'][idx][-1]) for idx in indexes]
        max_prev_posts_length = max(res['prev_posts_length'])
        res['prev_posts'] = np.zeros((batch_size, max_prev_posts_length), dtype=int)
        for i, idx in enumerate(indexes):
            prev_post = self.data[key]['posts'][idx][-1]
            res['prev_posts'][i, :len(prev_post)] = prev_post

        res['responses_length'] = [len(self.data[key]['responses'][idx]) for idx in indexes]
        max_responses_length = max(res['responses_length'])
        res['responses'] = np.zeros((batch_size, max_responses_length), dtype=int)
        for i, idx in enumerate(indexes):
            response = self.data[key]['responses'][idx]
            res['responses'][i, :len(response)] = response

        res['responses_allvocabs'] = res['responses'].copy()

        res['posts'][res['posts'] >= self.valid_vocab_len] = self.unk_id
        res['prev_posts'][res['prev_posts'] >= self.valid_vocab_len] = self.unk_id
        res['responses'][res['responses'] >= self.valid_vocab_len] = self.unk_id
        return res

    def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob", invalid_vocab=False):
        '''Get metrics for teacher-forcing.

        It contains:

        * :class:`.metric.PerplexityMetric`

        Arguments:
            gen_log_prob_key (str):  The key of predicted log probability over words.
                Refer to :class:`.metric.PerplexityMetric`. Default: ``gen_log_prob``.
            invalid_vocab (bool): Whether ``gen_log_prob`` contains invalid vocab.
                Refer to :class:`.metric.PerplexityMetric`. Default: ``False``.


        Returns:
            A :class:`.metric.MetricChain` object.
        '''
        metric = MetricChain()
        metric.add_metric(
            PerplexityMetric(self, reference_allvocabs_key="resp_allvocabs", reference_len_key="resp_length",
                             gen_log_prob_key=gen_log_prob_key, invalid_vocab=invalid_vocab))
        return metric

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


class MyMemHRED(MultiTurnDialog):
    def __init__(self, file_id="../data/film", min_vocab_times=0, max_sent_length=10086, invalid_vocab_times=0,
                 num_turns=8, max_know_length=100):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        self._num_turns = num_turns
        self._max_know_length = max_know_length
        super(MyMemHRED, self).__init__()


    def _load_data(self):
        r'''Loading dataset, invoked during the initialization of :class:`MultiTurnDialog`.
        '''
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        for key in self.key_name:
            origin_data[key] = {'posts': [], 'prev_posts': [], 'responses': [], 'kg': [], 'kg_index': []}
            datas = json.load(open('%s/%s.json' % (self._file_path, key), encoding='utf8'))

            for data in datas:
                messages = data['messages']
                turn = []
                kg = []
                kg_index = []
                kg_dict = {}
                for message in messages:
                    sent = jieba.lcut(message['message'])
                    turn.append(sent)

                    count_token(sent)

                    kg_index.append([])
                    if 'attrs' in message:
                        for attr in message['attrs']:
                            h = jieba.lcut(attr['zsname'].replace('【', '').replace('】', ''))
                            r = jieba.lcut(attr['zsattrname'].replace('【', '').replace('】', ''))
                            t = jieba.lcut(attr['zsattrvalue'].replace('【', '').replace('】', ''))
                            k = tuple((tuple(h), tuple(r), tuple(t)))
                            if k not in kg_dict:
                                kg_dict[k] = len(kg)
                                kg.append(k)
                            kg_index[-1].append(kg_dict[k])
                            count_token(h + r + t)


                for i in range(len(turn) - 1):
                    posts = [turn[j] for j in range(max(0, (i + 1) - (self._num_turns - 1)), i + 1)]
                    prev_post = posts[-1]
                    response = turn[i + 1]

                    origin_data[key]['posts'].append(posts)
                    origin_data[key]['prev_posts'].append(prev_post)
                    origin_data[key]['responses'].append(response)
                    origin_data[key]['kg'].append(kg)
                    origin_data[key]['kg_index'].append(kg_index[i + 1])

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
        line2id = lambda line: ([self.go_id] + list(map(lambda word: word2id.get(word, self.unk_id), line)) + \
                                [self.eos_id])[:self._max_sent_length]
        lines2id = lambda lines: list(map(line2id, lines))
        know2id = lambda line: list(map(lambda word: word2id.get(word, self.unk_id), line))
        knows2id = lambda lines: list(map(know2id, lines))

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}
            data[key]['posts'] = list(map(lines2id, origin_data[key]['posts']))
            data[key]['responses'] = list(map(line2id, origin_data[key]['responses']))
            data[key]['prev_posts'] = list(map(line2id, origin_data[key]['prev_posts']))

            # kg: 4-d (id, know_id, 3 triple, ele seq)
            # kg_index: 2-d (id, kg index)
            data[key]['kg'] = [list(map(knows2id, kg)) for kg in origin_data[key]['kg']]
            data[key]['kg_index'] = origin_data[key]['kg_index']

            data_size[key] = len(data[key]['responses'])

        return vocab_list, valid_vocab_len, data, data_size


    def get_batch(self, key, indexes):
        '''{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}

        Returns:
            (dict): A dict at least contains:
            {MultiTurnDialog.GET_BATCH_RETURNS_DICT}

            See the example belows.

        Examples:
            {MultiTurnDialog.GET_BATCH_EXAMPLES_PART}
            }
        '''
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        batch_size = len(indexes)
        res = {}

        res['context_length'] = [len(self.data[key]['posts'][idx]) for idx in indexes]
        max_context_length = max(res['context_length'])
        res['posts_length'] = np.zeros((batch_size * max_context_length), dtype=int)
        for i, idx in enumerate(indexes):
            posts_length = [len(each) for each in self.data[key]['posts'][idx]]
            res['posts_length'][i * max_context_length: i * max_context_length + len(posts_length)] = posts_length

        max_posts_length = np.max(res['posts_length'])
        res['posts'] = np.zeros((batch_size * max_context_length, max_posts_length), dtype=int)
        for i, idx in enumerate(indexes):
            for j in range(res['context_length'][i]):
                post = self.data[key]['posts'][idx][j]
                res['posts'][i * max_context_length + j, :len(post)] = post

        res['prev_posts_length'] = [len(self.data[key]['posts'][idx][-1]) for idx in indexes]
        max_prev_posts_length = max(res['prev_posts_length'])
        res['prev_posts'] = np.zeros((batch_size, max_prev_posts_length), dtype=int)
        for i, idx in enumerate(indexes):
            prev_post = self.data[key]['posts'][idx][-1]
            res['prev_posts'][i, :len(prev_post)] = prev_post

        res['responses_length'] = [len(self.data[key]['responses'][idx]) for idx in indexes]
        max_responses_length = max(res['responses_length'])
        res['responses'] = np.zeros((batch_size, max_responses_length), dtype=int)
        for i, idx in enumerate(indexes):
            response = self.data[key]['responses'][idx]
            res['responses'][i, :len(response)] = response

        res['responses_allvocabs'] = res['responses'].copy()

        res['posts'][res['posts'] >= self.valid_vocab_len] = self.unk_id
        res['prev_posts'][res['prev_posts'] >= self.valid_vocab_len] = self.unk_id
        res['responses'][res['responses'] >= self.valid_vocab_len] = self.unk_id

        max_kg_num = max([len(self.data[key]['kg'][idx]) for idx in indexes])
        res["kg_h_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res["kg_hr_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res['kg_hrt_length'] = np.zeros((batch_size, max_kg_num), dtype=int)
        for i, idx in enumerate(indexes):
            kg_h_length = [min(self._max_know_length, len(sent[0]))
                           for sent in self.data[key]['kg'][idx]]
            res["kg_h_length"][i, :len(kg_h_length)] = kg_h_length
            kg_hr_length = [min(self._max_know_length, len(sent[0]) + len(sent[1]))
                            for sent in self.data[key]['kg'][idx]]
            res["kg_hr_length"][i, :len(kg_hr_length)] = kg_hr_length
            kg_hrt_length = [min(self._max_know_length, len(sent[0]) + len(sent[1]) + len(sent[2]))
                             for sent in self.data[key]['kg'][idx]]
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

    def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob", invalid_vocab=False):
        '''Get metrics for teacher-forcing.

        It contains:

        * :class:`.metric.PerplexityMetric`

        Arguments:
            gen_log_prob_key (str):  The key of predicted log probability over words.
                Refer to :class:`.metric.PerplexityMetric`. Default: ``gen_log_prob``.
            invalid_vocab (bool): Whether ``gen_log_prob`` contains invalid vocab.
                Refer to :class:`.metric.PerplexityMetric`. Default: ``False``.


        Returns:
            A :class:`.metric.MetricChain` object.
        '''
        metric = MetricChain()
        metric.add_metric(
            PerplexityMetric(self, reference_allvocabs_key="resp_allvocabs", reference_len_key="resp_length",
                             gen_log_prob_key=gen_log_prob_key, invalid_vocab=invalid_vocab))
        return metric

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