import tensorflow as tf
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

            local_dis = tf.nn.log_softmax(tf.einsum('aij,kj->aik', outputs, weights) + bias)
            local_labels = tf.reshape(targets, [-1])
            local_masks = tf.reshape(masks, [-1])

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
