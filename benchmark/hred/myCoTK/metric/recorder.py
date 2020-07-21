r"""
Containing some recorders.
"""
import numpy as np
from cotk.metric import MetricBase

class SingleTurnResponseRecorder(MetricBase):
    _name = 'SingleTurnResponseRecorder'
    _version = 1
    def __init__(self, dataloader, resp_allvocabs_key="resp_allvocabs", gen_key="gen"):
        super().__init__(self._name, self._version)
        self.dataloader = dataloader
        self.resp_allvocabs_key = resp_allvocabs_key
        self.gen_key = gen_key
        self.post_list = []
        self.resp_list = []
        self.gen_list = []

    def forward(self, data):
        super().forward(data)
        resp_allvocabs = data[self.resp_allvocabs_key]
        gen = data[self.gen_key]

        if not isinstance(resp_allvocabs, (np.ndarray, list)):
            raise TypeError("Unknown type for resp_allvocabs")
        if not isinstance(gen, (np.ndarray, list)):
            raise TypeError("Unknown type for gen")

        if len(resp_allvocabs) != len(gen):
            raise ValueError("Batch num is not matched.")
        for i, resp_sen in enumerate(resp_allvocabs):
            self.resp_list.append(self.dataloader.convert_ids_to_tokens(resp_sen[1:]))
            self.gen_list.append(self.dataloader.convert_ids_to_tokens(gen[i]))

    def close(self):
        '''
        Returns:
            (dict): Return a dict which contains

            * **post**: a list of post sentences. A jagged 2-d array of int.
              Size:``[batch_size, ~sent_length]``, where "~" means different
              sizes in this dimension is allowed.
            * **resp**: a list of response sentences. A jagged 2-d array of int.
              Size:``[batch_size, ~sent_length]``, where "~" means different
              sizes in this dimension is allowed.
            * **gen**: A list of generated sentences. A jagged 2-d array of int.
              Size:``[batch_size, ~sent_length]``, where "~" means different
              sizes in this dimension is allowed.
        '''
        res = super().close()
        res.update({"resp": self.resp_list, "gen": self.gen_list})
        return res