import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.layers import utils
from tensorflow.python.layers.core import Dense

def output_projection_layer(num_units, num_symbols, num_samples=None, name="my_dense"):
    def sampled_sequence_loss(outputs, targets, masks):
        with variable_scope.variable_scope('decoder_rnn/%s' % name):
            weights = tf.transpose(tf.get_variable("kernel", [num_units, num_symbols]))
            bias = tf.get_variable("bias", [num_symbols])

            batch_size = tf.shape(targets)[0]

            local_labels = tf.reshape(targets, [-1])
            local_masks = tf.reshape(masks, [-1])
            local_dis = tf.nn.log_softmax(tf.einsum('aij,kj->aik', outputs, weights) + bias)
            #local_dis = tf.einsum('aij,kj->aik', outputs, weights) + bias

            y_prob = tf.reshape(local_dis, [-1, num_symbols])
            labels_onehot = tf.one_hot(local_labels, num_symbols)
            labels_onehot = tf.clip_by_value(labels_onehot, 0.0, 1.0)
            local_loss = tf.reduce_sum(-labels_onehot * y_prob, 1) * local_masks

            loss = tf.reduce_sum(local_loss)
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights

            reshape_loss = tf.reduce_sum(tf.reshape(local_loss, [batch_size, -1]), axis=1)
            reshape_mask = tf.reduce_sum(tf.reshape(local_masks, [batch_size, -1]), axis=1)

            return local_dis, loss / total_size, tf.div(reshape_loss, reshape_mask)
    
    return sampled_sequence_loss

# You can customize the output_projection by overriding Dense
class MyDense(Dense):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(MyDense, self).__init__(units=units,
                                activation=activation,
                                use_bias=use_bias,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                activity_regularizer=activity_regularizer,
                                kernel_constraint=kernel_constraint,
                                bias_constraint=bias_constraint,
                                trainable=trainable,
                                name=name,
                                **kwargs)

class MyAttention(tf.contrib.seq2seq.BahdanauAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_mask=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 custom_key_value_fn=None,
                 name="MyBahdanauAttention"):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.    This
                tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length: (optional) Sequence lengths for the batch entries
                in memory.    If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
            normalize: Python boolean.    Whether to normalize the energy term.
            probability_fn: (optional) A `callable`.    Converts the score to
                probabilities.    The default is `tf.nn.softmax`. Other options include
                `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
                Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            dtype: The data type for the query and memory layers of the attention
                mechanism.
            custom_key_value_fn: (optional): The custom function for
                computing keys and values.
            name: Name to use when creating ops.
        """
        super(MyAttention, self).__init__(num_units,
                                        memory,
                                        name=name)
        self.memory_sequence_mask = memory_sequence_mask

    def __call__(self, query, state):
        with tf.variable_scope(None, "bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            attention_v = tf.get_variable(
                    "attention_v", [self._num_units], dtype=query.dtype)
            if not self._normalize:
                attention_g = None
                attention_b = None
            else:
                attention_g = tf.get_variable(
                        "attention_g",
                        dtype=query.dtype,
                        initializer=tf.constant_initializer(
                                tf.sqrt((1. / self._num_units))),
                        shape=())
                attention_b = tf.get_variable(
                        "attention_b", [self._num_units],
                        dtype=query.dtype,
                        initializer=tf.zeros_initializer())

            score = self._bahdanau_score(
                    processed_query,
                    self._keys,
                    attention_v,
                    attention_g=attention_g,
                    attention_b=attention_b)
        #alignments = self._probability_fn(score, state)
        mask_value = tf.dtypes.as_dtype(processed_query.dtype).as_numpy_dtype(-np.inf)
        alignments = tf.nn.softmax(tf.where(self.memory_sequence_mask, score, tf.ones_like(score) * mask_value))
        next_state = alignments
        return alignments, next_state

    def _bahdanau_score(self,
                    processed_query,
                    keys,
                    attention_v,
                    attention_g=None,
                    attention_b=None):
        processed_query = tf.expand_dims(processed_query, 1)
        if attention_g is not None and attention_b is not None:
            normed_v = attention_g * attention_v * tf.rsqrt(
                    tf.reduce_sum(tf.square(attention_v)))
            return tf.reduce_sum(
                    normed_v * tf.tanh(keys + processed_query + attention_b), [2])
        else:
            return tf.reduce_sum(
                    attention_v * tf.tanh(keys + processed_query), [2])
