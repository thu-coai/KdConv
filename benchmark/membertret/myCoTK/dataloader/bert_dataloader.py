'''
A module for BERT dataloader
'''
from cotk.dataloader import BERTLanguageProcessingBase
from cotk._utils import trim_before_target
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
import tqdm
import time
from itertools import chain
from collections import Counter
import json
import random

from cotk.metric import MetricChain, SingleTurnDialogRecorder
from ..metric import BleuCorpusMetric, SingleTurnDistinct
from pytorch_pretrained_bert.tokenization import BertTokenizer

import jieba
from gensim.summarization import bm25


class MyBERTRetrieval(BERTLanguageProcessingBase):
    def __init__(self, file_id, bert_vocab_name, do_lower_case, num_choices=10,
                 max_sent_length=192, num_turns=8,
                 ext_vocab=None, key_name=None, cpu_count=None):
        self.ext_vocab = ext_vocab or ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        self._max_sent_length = max_sent_length
        self._file_path = file_id
        self.num_choices = num_choices
        self.num_turns = num_turns
        super().__init__(self.ext_vocab, key_name, bert_vocab_name)

        self.tokenizer = BertTokenizer(vocab_file=bert_vocab_name, do_lower_case=do_lower_case)
        self._build_bert_vocab()

        if cpu_count is not None:
            self.cpu_count = cpu_count
        elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
            self.cpu_count = int(os.environ["CPU_COUNT"])
        else:
            self.cpu_count = multiprocessing.cpu_count()

    def _load_data(self):
        r'''Loading dataset, invoked by `BERTLanguageProcessingBase.__init__`
        '''
        print('begin load data...')
        begin_time = time.time()
        origin_data = {}

        with open("../data/resources/chinese_stop_words.txt") as f:
            stop_words = set([w.strip() for w in f.readlines()])

        for key in self.key_name:
            corpus_resp = []
            corpus_post = []
            origin_data[key] = {'resp': [], 'post_bert': [], 'resp_bert': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key)))

            for data in datas:
                messages = data['messages']
                i = 0

                post_sent = []
                corpus_post_sent = []
                while i + 1 < len(messages):
                    if i == 0:
                        tmp_sent = self.convert_tokens_to_bert_ids(self.tokenize(messages[i]['message']))
                        post_sent = [tmp_sent]
                        corpus_post_sent = [
                            [token for token in jieba.lcut(messages[i]['message']) if token not in stop_words]]

                    origin_data[key]['resp'].append(messages[i + 1]['message'])
                    sentence = [token for token in jieba.lcut(messages[i + 1]['message']) if token not in stop_words]
                    nxt_sent = self.convert_tokens_to_bert_ids(self.tokenize(messages[i + 1]['message']))

                    origin_data[key]['post_bert'].append(post_sent)
                    origin_data[key]['resp_bert'].append(nxt_sent)
                    corpus_resp.append(sentence)
                    corpus_post.append(corpus_post_sent)

                    post_sent = (post_sent + [nxt_sent])[- self.num_turns + 1:]
                    corpus_post_sent = corpus_post_sent + [sentence]

                    i += 1

            distractor_file = os.path.join(self._file_path, '%s_distractors.json' % key)
            if os.path.exists(distractor_file):
                with open(distractor_file) as f:
                    origin_data[key]['resp_distractors'] = json.load(f)
                origin_data[key]['resp_distractors_bert'] = [
                    [self.convert_tokens_to_bert_ids(self.tokenize(sent)) for sent in distractors] for distractors in
                    origin_data[key]['resp_distractors']]

            else:
                bm25Model = bm25.BM25(corpus_resp)
                origin_data[key]['resp_distractors'] = []
                origin_data[key]['resp_distractors_bert'] = []

                for idx in range(len(corpus_resp)):
                    posts = corpus_post[idx]
                    resp = corpus_resp[idx]

                    count_post_token = {}
                    for token in list(chain(*posts)):
                        if token in count_post_token:
                            count_post_token[token] += 1
                        else:
                            count_post_token[token] = 1

                    token_tfidf = {}
                    for token in count_post_token:
                        if token in bm25Model.idf:
                            token_tfidf[token] = count_post_token[token] * bm25Model.idf[token]

                    token_tfidf = sorted(list(token_tfidf.items()), key=lambda x: (-x[1], x[0]))
                    top5 = [each[0] for each in token_tfidf[:5]]

                    bm_scores = bm25Model.get_scores(list(set(resp + top5)))
                    bm_scores = np.array(bm_scores)
                    rank = np.argsort(bm_scores).tolist()
                    if idx in rank[-self.num_choices:]:
                        idxs = [each for each in rank[-self.num_choices:] if each != idx]
                    else:
                        idxs = rank[-self.num_choices + 1:]

                    origin_data[key]['resp_distractors'].append([origin_data[key]['resp'][k] for k in idxs])
                    origin_data[key]['resp_distractors_bert'].append([origin_data[key]['resp_bert'][k] for k in idxs])

                with open(distractor_file, 'w') as f:
                    json.dump(origin_data[key]['resp_distractors'], f, ensure_ascii=False, indent=4)

        print('finish tokenizing sentences...%f' % (time.time() - begin_time))
        vocab_list = [each for each in self.bert_id2word]
        valid_vocab_len = len(vocab_list)

        print("vocab list length = %d" % len(vocab_list))

        data_size = {key: len(origin_data[key]['resp']) for key in self.key_name}
        return vocab_list, valid_vocab_len, origin_data, data_size

    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {
            'resp': [self.data[key]['resp'][i] for i in indexes], # original response
            'can_resps': [],
            'input_ids': None,
            'input_mask': None,
            'segment_ids': None,
            'labels': None
        }
        batch_size = len(indexes)
        labels = np.zeros((batch_size * self.num_choices), dtype=int)

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []

        for iidx, idx in enumerate(indexes):
            post_bert = self.data[key]['post_bert'][idx] # a list of historical utterances
            resp_bert = self.data[key]['resp_bert'][idx]
            resp_distractors_bert = self.data[key]['resp_distractors_bert'][idx]

            ori_can_resps = [self.data[key]['resp'][idx]] + self.data[key]['resp_distractors'][idx]
            can_resps = []

            options = [resp_bert] + resp_distractors_bert
            options = list((ii, option) for ii, option in enumerate(options))
            random.shuffle(options)
            option_ids = [each[0] for each in options]
            options = [each[1] for each in options]

            resp_max_length = max([len(each) for each in options])
            post_max_length = self._max_sent_length - 3 - resp_max_length
            post_bert = list(chain(*post_bert))[-post_max_length:]
            for tt, (option_id, option) in enumerate(zip(option_ids, options)):
                can_resps.append(ori_can_resps[option_id])

                input_id = [self.bert_go_id] + post_bert + [self.bert_eos_id] + option + [self.bert_eos_id]
                input_mask = [1] * len(input_id)
                segment_ids = [0] * (len(post_bert) + 2) + [1] * (len(option) + 1)

                assert len(input_id) == len(segment_ids)

                padding = [0] * (self._max_sent_length - len(input_id))
                input_id = input_id + padding
                input_mask = input_mask + padding
                segment_ids = segment_ids + padding

                all_input_ids.append(input_id)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
                labels[iidx * self.num_choices + tt] = 1 if option_id == 0 else 0

            res['can_resps'].append(can_resps)

        assert len(all_input_ids) == batch_size * self.num_choices

        res['input_ids'] = all_input_ids
        res['input_mask'] = all_input_mask
        res['segment_ids'] = all_segment_ids
        res['labels'] = labels

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
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, \
            reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric


class MyMemBERTRetrieval(BERTLanguageProcessingBase):
    def __init__(self, file_id, bert_vocab_name, do_lower_case, num_choices=10,
                 max_sent_length=192, max_know_length=100, num_turns=8,
                 ext_vocab=None, key_name=None, cpu_count=None):
        self.ext_vocab = ext_vocab or ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        self._max_sent_length = max_sent_length
        self._max_know_length = max_know_length
        self._file_path = file_id
        self.num_choices = num_choices
        self.num_turns = num_turns
        super().__init__(self.ext_vocab, key_name, bert_vocab_name)

        self.tokenizer = BertTokenizer(vocab_file=bert_vocab_name, do_lower_case=do_lower_case)
        self._build_bert_vocab()

        if cpu_count is not None:
            self.cpu_count = cpu_count
        elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
            self.cpu_count = int(os.environ["CPU_COUNT"])
        else:
            self.cpu_count = multiprocessing.cpu_count()

    def _load_data(self):
        r'''Loading dataset, invoked by `BERTLanguageProcessingBase.__init__`
        '''
        print('begin load data...')
        begin_time = time.time()
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        with open("../data/resources/chinese_stop_words.txt") as f:
            stop_words = set([w.strip() for w in f.readlines()])

        for key in self.key_name:
            corpus_resp = []
            corpus_post = []
            origin_data[key] = {'kg_index': [], 'kg': [], 'resp': [], 'post_bert': [], 'resp_bert': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key)))

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
                
                i = 0
                post_sent = []
                corpus_post_sent = []
                while i + 1 < len(messages):
                    if i == 0:
                        tmp_sent = self.convert_tokens_to_bert_ids(self.tokenize(messages[i]['message']))
                        post_sent = [tmp_sent]
                        corpus_post_sent = [
                            [token for token in jieba.lcut(messages[i]['message']) if token not in stop_words]]

                    origin_data[key]['resp'].append(messages[i + 1]['message'])
                    sentence = [token for token in jieba.lcut(messages[i + 1]['message']) if token not in stop_words]
                    nxt_sent = self.convert_tokens_to_bert_ids(self.tokenize(messages[i + 1]['message']))

                    origin_data[key]['post_bert'].append(post_sent)
                    origin_data[key]['resp_bert'].append(nxt_sent)
                    corpus_resp.append(sentence)
                    corpus_post.append(corpus_post_sent)

                    post_sent = (post_sent + [nxt_sent])[- self.num_turns + 1:]
                    corpus_post_sent = corpus_post_sent + [sentence]
                    origin_data[key]['kg'].append(kg)
                    origin_data[key]['kg_index'].append(kg_index[i + 1])

                    i += 1

            distractor_file = os.path.join(self._file_path, '%s_distractors.json' % key)
            if os.path.exists(distractor_file):
                with open(distractor_file) as f:
                    origin_data[key]['resp_distractors'] = json.load(f)
                origin_data[key]['resp_distractors_bert'] = [
                    [self.convert_tokens_to_bert_ids(self.tokenize(sent)) for sent in distractors] for distractors in
                    origin_data[key]['resp_distractors']]

            else:
                bm25Model = bm25.BM25(corpus_resp)
                origin_data[key]['resp_distractors'] = []
                origin_data[key]['resp_distractors_bert'] = []

                for idx in range(len(corpus_resp)):
                    posts = corpus_post[idx]
                    resp = corpus_resp[idx]

                    count_post_token = {}
                    for token in list(chain(*posts)):
                        if token in count_post_token:
                            count_post_token[token] += 1
                        else:
                            count_post_token[token] = 1

                    token_tfidf = {}
                    for token in count_post_token:
                        if token in bm25Model.idf:
                            token_tfidf[token] = count_post_token[token] * bm25Model.idf[token]

                    token_tfidf = sorted(list(token_tfidf.items()), key=lambda x: (-x[1], x[0]))
                    top5 = [each[0] for each in token_tfidf[:5]]

                    bm_scores = bm25Model.get_scores(list(set(resp + top5)))
                    bm_scores = np.array(bm_scores)
                    rank = np.argsort(bm_scores).tolist()
                    if idx in rank[-self.num_choices:]:
                        idxs = [each for each in rank[-self.num_choices:] if each != idx]
                    else:
                        idxs = rank[-self.num_choices + 1:]

                    origin_data[key]['resp_distractors'].append([origin_data[key]['resp'][k] for k in idxs])
                    origin_data[key]['resp_distractors_bert'].append([origin_data[key]['resp_bert'][k] for k in idxs])

                with open(distractor_file, 'w') as f:
                    json.dump(origin_data[key]['resp_distractors'], f, ensure_ascii=False, indent=4)

        print('finish tokenizing sentences...%f' % (time.time() - begin_time))
        vocab_list = [each for each in self.bert_id2word]
        valid_vocab_len = len(vocab_list)
        print("bert vocab list length = %d" % len(vocab_list))

        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        self.id2know_word = list(map(lambda x: x[0], vocab))
        self.know_word2id = {w: i for i, w in enumerate(self.id2know_word)}
        print("knowledge vocab list length = %d" % len(self.id2know_word))

        know2id = lambda line: list(map(lambda word: self.know_word2id.get(word, self.unk_id), line))
        knows2id = lambda lines: list(map(know2id, lines))
        for key in self.key_name:
            origin_data[key]['kg'] = [list(map(knows2id, kg)) for kg in origin_data[key]['kg']]

        data_size = {key: len(origin_data[key]['resp']) for key in self.key_name}
        return vocab_list, valid_vocab_len, origin_data, data_size

    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {
            'resp': [self.data[key]['resp'][i] for i in indexes], # original response
            'can_resps': [],
            'input_ids': None,
            'input_mask': None,
            'segment_ids': None,
            'resp_ids': None,
            'resp_mask': None,
            'labels': None
        }
        batch_size = len(indexes)
        labels = np.zeros((batch_size * self.num_choices), dtype=int)

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        resp_ids = []
        resp_mask = []
        single_resp_max_length = min(
            max([len(self.data[key]['resp_bert'][idx]) for idx in indexes] +
                [max([len(distractor) for distractor in self.data[key]['resp_distractors_bert'][idx]]) for idx in indexes]) - 2,
            self._max_sent_length - 2)

        for iidx, idx in enumerate(indexes):
            post_bert = self.data[key]['post_bert'][idx] # a list of historical utterances
            resp_bert = self.data[key]['resp_bert'][idx]
            resp_distractors_bert = self.data[key]['resp_distractors_bert'][idx]

            ori_can_resps = [self.data[key]['resp'][idx]] + self.data[key]['resp_distractors'][idx]
            can_resps = []

            options = [resp_bert] + resp_distractors_bert
            options = list((ii, option) for ii, option in enumerate(options))
            random.shuffle(options)
            option_ids = [each[0] for each in options]
            options = [each[1] for each in options]

            resp_max_length = max([len(each) for each in options])
            post_max_length = self._max_sent_length - 3 - resp_max_length
            post_bert = list(chain(*post_bert))[-post_max_length:]

            for tt, (option_id, option) in enumerate(zip(option_ids, options)):
                can_resps.append(ori_can_resps[option_id])

                resp_id = [self.bert_go_id] + option[: single_resp_max_length] + [self.bert_eos_id]
                resp_msk = [1] * len(resp_id)
                padding = [0] * (single_resp_max_length + 2 - len(resp_id))
                resp_ids.append(resp_id + padding)
                resp_mask.append(resp_msk + padding)

                input_id = [self.bert_go_id] + post_bert + [self.bert_eos_id] + option + [self.bert_eos_id]
                input_mask = [1] * len(input_id)
                segment_ids = [0] * (len(post_bert) + 2) + [1] * (len(option) + 1)

                assert len(input_id) == len(segment_ids)

                padding = [0] * (self._max_sent_length - len(input_id))
                input_id = input_id + padding
                input_mask = input_mask + padding
                segment_ids = segment_ids + padding

                all_input_ids.append(input_id)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
                labels[iidx * self.num_choices + tt] = 1 if option_id == 0 else 0

            res['can_resps'].append(can_resps)

        assert len(all_input_ids) == batch_size * self.num_choices

        res['input_ids'] = all_input_ids
        res['input_mask'] = all_input_mask
        res['segment_ids'] = all_segment_ids
        res['labels'] = labels
        res['resp_ids'] = resp_ids
        res['resp_mask'] = resp_mask


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


        for k in ['kg_h_length', 'kg_hr_length', 'kg_hrt_length', 'kg', 'kg_index']:
            res[k] = res[k].repeat(self.num_choices, 0)


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
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, \
            reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric
