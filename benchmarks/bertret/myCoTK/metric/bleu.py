import random
import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from cotk.metric import MetricBase
from cotk.metric.bleu import _replace_unk
from cotk._utils import hooks


class BleuCorpusMetric(MetricBase):
    '''Metric for calculating BLEU.

    Arguments:
        {MetricBase.DATALOADER_ARGUMENTS}
        {MetricBase.REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
        {MetricBase.GEN_KEY_ARGUMENTS}
    '''

    _name = 'BleuCorpusMetric'
    _version = 1

    @hooks.hook_metric
    def __init__(self, dataloader, ignore_smoothing_error=False,\
            reference_allvocabs_key="ref_allvocabs", gen_key="gen"):
        super().__init__(self._name, self._version)
        self.dataloader = dataloader
        self.ignore_smoothing_error = ignore_smoothing_error
        self.reference_allvocabs_key = reference_allvocabs_key
        self.gen_key = gen_key
        self.refs = []
        self.hyps = []

    def forward(self, data):
        '''Processing a batch of data.

        Arguments:
            data (dict): A dict at least contains the following keys:

                {MetricBase.FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS}
                {MetricBase.FORWARD_GEN_ARGUMENTS}

                Here is an example for data:
                    >>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
                    >>> #   "been", "to", "China"]
                    >>> data = {
                    ...     reference_allvocabs_key: [[2,4,3], [2,5,6,3]]
                    ...        gen_key: [[4,5,3], [6,7,8,3]]
                    ... }
        '''
        super().forward(data)
        gen = data[self.gen_key]
        resp = data[self.reference_allvocabs_key]

        if not isinstance(gen, (np.ndarray, list)):
            raise TypeError("Unknown type for gen.")
        if not isinstance(resp, (np.ndarray, list)):
            raise TypeError("Unknown type for resp")

        if len(resp) != len(gen):
            raise ValueError("Batch num is not matched.")

        relevant_data = []
        for gen_sen, resp_sen in zip(gen, resp):
            self.hyps.append(self.dataloader.trim(gen_sen))
            reference = list(self.dataloader.trim(resp_sen[1:]))
            relevant_data.append(reference)
            self.refs.append([reference])
        self._hash_relevant_data(relevant_data)

    @hooks.hook_metric_close
    def close(self):
        '''
        Returns:
            (dict): Return a dict which contains

            * **bleu**: bleu value.
            * **bleu hashvalue**: hash value for bleu metric, same hash value stands
              for same evaluation settings.
        '''
        result = super().close()
        if (not self.hyps) or (not self.refs):
            raise RuntimeError("The metric has not been forwarded data correctly.")

        self.hyps = _replace_unk(self.hyps, self.dataloader.unk_id)

        for i in range(1, 5):
            try:
                weights = [1 / i] * i + [0] * (4 - i)
                result.update({"bleu-%d" % i: corpus_bleu(self.refs, self.hyps, weights,
                                                          smoothing_function=SmoothingFunction().method3)})

            except ZeroDivisionError as _:
                if not self.ignore_smoothing_error:
                    raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
                    usually caused when there is only one sample and the sample length is 1.")
                result.update({"bleu-%d" % i: 0, "bleu hashvalue": self._hashvalue()})

        return result