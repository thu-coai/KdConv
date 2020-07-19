import numpy as np
import tensorflow as tf
import time
from itertools import chain

from tensorflow.python.ops.nn import dynamic_rnn
from utils.output_projection import output_projection_layer, MyDense, MyAttention
from utils import SummaryHelper

import os
import jieba
import json

class HredModel(object):
	def __init__(self, data, args, embed):
		#self.init_states = tf.placeholder(tf.float32, (None, args.ch_size), 'ctx_inps')  # batch*ch_size
		self.posts = tf.placeholder(tf.int32, (None, None), 'enc_inps')  # batch * num_turns-1 * len
		self.posts_length = tf.placeholder(tf.int32, (None,), 'enc_lens')  # batch * num_turns-1

		self.prev_posts = tf.placeholder(tf.int32, (None, None), 'enc_prev_inps')
		self.prev_posts_length = tf.placeholder(tf.int32, (None,), 'enc_prev_lens')

		self.origin_responses = tf.placeholder(tf.int32, (None, None), 'dec_inps')  # batch*len
		self.origin_responses_length = tf.placeholder(tf.int32, (None,), 'dec_lens')  # batch
		self.context_length = tf.placeholder(tf.int32, (None,), 'ctx_lens')
		self.is_train = tf.placeholder(tf.bool)

		num_past_turns = tf.shape(self.posts)[0] // tf.shape(self.origin_responses)[0]

		# deal with original data to adapt encoder and decoder
		batch_size, decoder_len = tf.shape(self.origin_responses)[0], tf.shape(self.origin_responses)[1]
		self.responses = tf.split(self.origin_responses, [1, decoder_len - 1], 1)[1]  # no go_id
		self.responses_length = self.origin_responses_length - 1
		self.responses_input = tf.split(self.origin_responses, [decoder_len - 1, 1], 1)[0]  # no eos_id
		self.responses_target = self.responses
		decoder_len = decoder_len - 1
		self.posts_input = self.posts  # batch*len
		self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length - 1, decoder_len), reverse=True, axis=1),
									   [-1, decoder_len])

		# initialize the training process
		self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * args.lr_decay)
		self.global_step = tf.Variable(0, trainable=False)

		# build the embedding table and embedding input
		if embed is None:
			# initialize the embedding randomly
			self.embed = tf.get_variable('embed', [data.vocab_size, args.embedding_size], tf.float32)
		else:
			# initialize the embedding by pre-trained word vectors
			self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

		self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts)
		self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)
		# self.encoder_input = tf.cond(self.is_train,
		#							 lambda: tf.nn.dropout(tf.nn.embedding_lookup(self.embed, self.posts_input), 0.8),
		#							 lambda: tf.nn.embedding_lookup(self.embed, self.posts_input))  # batch*len*unit
		# self.decoder_input = tf.cond(self.is_train,
		#							 lambda: tf.nn.dropout(tf.nn.embedding_lookup(self.embed, self.responses_input), 0.8),
		#							 lambda: tf.nn.embedding_lookup(self.embed, self.responses_input))

		# build rnn_cell
		cell_enc = tf.nn.rnn_cell.GRUCell(args.eh_size)
		cell_ctx = tf.nn.rnn_cell.GRUCell(args.ch_size)
		cell_dec = tf.nn.rnn_cell.GRUCell(args.dh_size)

		# build encoder
		with tf.variable_scope('encoder'):
			encoder_output, encoder_state = dynamic_rnn(cell_enc, self.encoder_input, self.posts_length, dtype=tf.float32,
														scope="encoder_rnn")

		with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
			prev_output, _ = dynamic_rnn(cell_enc, tf.nn.embedding_lookup(self.embed, self.prev_posts), self.prev_posts_length,
										 dtype=tf.float32, scope="encoder_rnn")

		# encoder_hidden_size = tf.shape(encoder_state)[-1]

		with tf.variable_scope('context'):
			encoder_state_reshape = tf.reshape(encoder_state, [-1, num_past_turns, args.eh_size])
			context_output, self.context_state = dynamic_rnn(cell_ctx, encoder_state_reshape, self.context_length,
															 dtype=tf.float32, scope='context_rnn')

		# get output projection function
		output_fn = MyDense(data.vocab_size, use_bias=True)
		sampled_sequence_loss = output_projection_layer(args.dh_size, data.vocab_size, args.softmax_samples)

		# construct helper and attention
		train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_input, tf.maximum(self.responses_length, 1))
		infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed, tf.fill([batch_size], data.go_id), data.eos_id)

		#encoder_len = tf.shape(encoder_output)[1]
		#attention_memory = tf.reshape(encoder_output, [batch_size, -1, args.eh_size])
		#attention_mask = tf.reshape(tf.sequence_mask(self.posts_length, encoder_len), [batch_size, -1])
		'''
        attention_memory = context_output
        attention_mask = tf.reshape(tf.sequence_mask(self.context_length, self.num_turns - 1), [batch_size, -1])
        '''
		#attention_mask = tf.concat([tf.ones([batch_size, 1], tf.bool), attention_mask[:, 1:]], axis=1)
		#attn_mechanism = MyAttention(args.dh_size, attention_memory, attention_mask)
		attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(args.dh_size, prev_output,
				memory_sequence_length=tf.maximum(self.prev_posts_length, 1))
		cell_dec_attn = tf.contrib.seq2seq.AttentionWrapper(cell_dec, attn_mechanism, attention_layer_size=args.dh_size)
		ctx_state_shaping = tf.layers.dense(self.context_state, args.dh_size, activation=None)
		dec_start = cell_dec_attn.zero_state(batch_size, dtype=tf.float32).clone(cell_state=ctx_state_shaping)

		# build decoder (train)
		with tf.variable_scope('decoder'):
			decoder_train = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, train_helper, dec_start)
			train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train, impute_finished=True, scope="decoder_rnn")
			self.decoder_output = train_outputs.rnn_output
			# self.decoder_output = tf.nn.dropout(self.decoder_output, 0.8)
			self.decoder_distribution_teacher, self.decoder_loss, self.decoder_all_loss = \
				sampled_sequence_loss(self.decoder_output, self.responses_target, self.decoder_mask)

		# build decoder (test)
		with tf.variable_scope('decoder', reuse=True):
			decoder_infer = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, infer_helper, dec_start, output_layer=output_fn)
			infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished=True,
																	maximum_iterations=args.max_sent_length, scope="decoder_rnn")
			self.decoder_distribution = infer_outputs.rnn_output
			self.generation_index = tf.argmax(tf.split(self.decoder_distribution, [2, data.vocab_size - 2], 2)[1],
											  2) + 2  # for removing UNK

		# calculate the gradient of parameters and update
		self.params = [k for k in tf.trainable_variables() if args.name in k.name]
		opt = tf.train.AdamOptimizer(self.learning_rate)
		gradients = tf.gradients(self.decoder_loss, self.params)
		clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, args.grad_clip)
		self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

		# save checkpoint
		self.latest_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=args.checkpoint_max_to_keep,
										   pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
		self.best_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1, pad_step_number=True,
										 keep_checkpoint_every_n_hours=1.0)

		# create summary for tensorboard
		self.create_summary(args)

	def store_checkpoint(self, sess, path, key, name):
		if key == "latest":
			self.latest_saver.save(sess, path, global_step=self.global_step, latest_filename=name)
		else:
			self.best_saver.save(sess, path, global_step=self.global_step, latest_filename=name)

	def create_summary(self, args):
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), args)

		self.trainSummary = self.summaryHelper.addGroup(scalar=["loss", "perplexity"], prefix="train")

		scalarlist = ["loss", "perplexity"]
		tensorlist = []
		textlist = []
		emblist = []
		for i in args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
				embedding=emblist, prefix="dev")
		self.testSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
				embedding=emblist, prefix="test")


	def print_parameters(self):
		for item in self.params:
			print('%s: %s' % (item.name, item.get_shape()))

	def step_decoder(self, sess, data, forward_only=False, inference=False):
		input_feed = {
			self.posts: data['posts'],
			self.posts_length: data['posts_length'],
			self.origin_responses: data['responses'],
			self.origin_responses_length: data['responses_length'],
			self.context_length: data['context_length'],
			self.prev_posts: data['prev_posts'],
			self.prev_posts_length: data['prev_posts_length']
		}

		if inference:
			input_feed.update({self.is_train: False})
			output_feed = [self.generation_index, self.decoder_distribution_teacher, self.decoder_all_loss]
		else:
			input_feed.update({self.is_train: True})
			if forward_only:
				output_feed = [self.decoder_loss, self.decoder_distribution_teacher]
			else:
				output_feed = [self.decoder_loss, self.gradient_norm, self.update]

		return sess.run(output_feed, input_feed)


	def evaluate(self, sess, data, batch_size, key_name):
		loss = np.zeros((1,))
		times = 0
		data.restart(key_name, batch_size=batch_size, shuffle=False)
		batched_data = data.get_next_batch(key_name)
		while batched_data != None:
			outputs = self.step_decoder(sess, batched_data, forward_only=True)
			loss += outputs[0]
			times += 1
			batched_data = data.get_next_batch(key_name)
		loss /= times

		print('    perplexity on %s set: %.2f' % (key_name, np.exp(loss)))
		print(loss)
		return loss


	def train_process(self, sess, data, args):
		loss_step, time_step, epoch_step = np.zeros((1,)), .0, 0
		previous_losses = [1e18] * 3
		best_valid = 1e18
		data.restart("train", batch_size=args.batch_size, shuffle=True)
		batched_data = data.get_next_batch("train")
		for epoch_step in range(args.epochs):
			while batched_data != None:
				if self.global_step.eval() % args.checkpoint_steps == 0 and self.global_step.eval() != 0:
					show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
					print("Epoch %d global step %d learning rate %.4f step-time %.2f perplexity %s" % (
					epoch_step, self.global_step.eval(), self.learning_rate.eval(), time_step, show(np.exp(loss_step))))
					self.trainSummary(self.global_step.eval() // args.checkpoint_steps,
									  {'loss': loss_step, 'perplexity': np.exp(loss_step)})
					self.store_checkpoint(sess, '%s/checkpoint_latest/%s' % (args.model_dir, args.name), "latest",
										  args.name)

					dev_loss = self.evaluate(sess, data, args.batch_size, "dev")
					self.devSummary(self.global_step.eval() // args.checkpoint_steps,
									{'loss': dev_loss, 'perplexity': np.exp(dev_loss)})


					if np.sum(loss_step) > max(previous_losses):
						sess.run(self.learning_rate_decay_op)
					if dev_loss < best_valid:
						best_valid = dev_loss
						self.store_checkpoint(sess, '%s/checkpoint_best/%s' % (args.model_dir, args.name), "best",
											  args.name)

					previous_losses = previous_losses[1:] + [np.sum(loss_step)]
					loss_step, time_step = np.zeros((1,)), .0

				start_time = time.time()
				loss_step += self.step_decoder(sess, batched_data)[0] / args.checkpoint_steps
				time_step += (time.time() - start_time) / args.checkpoint_steps
				batched_data = data.get_next_batch("train")

			data.restart("train", batch_size=args.batch_size, shuffle=True)
			batched_data = data.get_next_batch("train")


	def test_process_hits(self, sess, data, args):

		with open(os.path.join(args.datapath, 'test_distractors.json'), 'r') as f:
			test_distractors = json.load(f)

		data.restart("test", batch_size=1, shuffle=False)
		batched_data = data.get_next_batch("test")

		loss_record = []
		cnt = 0
		while batched_data != None:

			for key in batched_data:
				if isinstance(batched_data[key], np.ndarray):
					batched_data[key] = batched_data[key].tolist()

			batched_data['responses_length'] = [len(batched_data['responses'][0])]
			batched_data['responses'] = batched_data['responses']
			for each_resp in test_distractors[cnt]:
				batched_data['responses'].append(
					[data.go_id] + data.convert_tokens_to_ids(jieba.lcut(each_resp)) + [data.eos_id])
				batched_data['responses_length'].append(len(batched_data['responses'][-1]))
			max_length = max(batched_data['responses_length'])
			resp = np.zeros((len(batched_data['responses']), max_length), dtype=int)
			for i, each_resp in enumerate(batched_data['responses']):
				resp[i, :len(each_resp)] = each_resp
			batched_data['responses'] = resp

			posts = []
			posts_length = []
			prev_posts = []
			prev_posts_length = []
			context_length = []
			for _ in range(len(resp)):
				posts += batched_data['posts']
				posts_length += batched_data['posts_length']
				prev_posts += batched_data['prev_posts']
				prev_posts_length += batched_data['prev_posts_length']
				context_length += batched_data['context_length']
			batched_data['posts'] = posts
			batched_data['posts_length'] = posts_length
			batched_data['prev_posts'] = prev_posts
			batched_data['prev_posts_length'] = prev_posts_length
			batched_data['context_length'] = context_length

			_, _, loss = self.step_decoder(sess, batched_data, inference=True)
			loss_record.append(loss)
			cnt += 1
			batched_data = data.get_next_batch("test")

		assert cnt == len(test_distractors)

		loss = np.array(loss_record)
		loss_rank = np.argsort(loss, axis=1)
		hits1 = float(np.mean(loss_rank[:, 0] == 0))
		hits3 = float(np.mean(np.min(loss_rank[:, :3], axis=1) == 0))
		return {'hits@1': hits1, 'hits@3': hits3}


	def test_process(self, sess, data, args):

		metric1 = data.get_teacher_forcing_metric()
		metric2 = data.get_inference_metric()
		data.restart("test", batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch("test")

		while batched_data != None:
			batched_responses_id, gen_log_prob, _ = self.step_decoder(sess, batched_data, inference=True)
			metric1_data = {'resp_allvocabs': np.array(batched_data['responses_allvocabs']),
							'resp_length': np.array(batched_data['responses_length']),
							'gen_log_prob': np.array(gen_log_prob)}
			metric1.forward(metric1_data)
			batch_results = []
			for response_id in batched_responses_id:
				response_id_list = response_id.tolist()
				if data.eos_id in response_id_list:
					result_id = response_id_list[:response_id_list.index(data.eos_id) + 1]
				else:
					result_id = response_id_list
				batch_results.append(result_id)

			metric2_data = {'gen': np.array(batch_results),
							'resp_allvocabs': np.array(batched_data['responses_allvocabs'])
							}
			metric2.forward(metric2_data)
			batched_data = data.get_next_batch("test")

		res = metric1.close()
		res.update(metric2.close())
		res.update(self.test_process_hits(sess, data, args))

		test_file = args.out_dir + "/%s_%s.txt" % (args.name, "test")
		with open(test_file, 'w') as f:
			print("Test Result:")
			res_print = list(res.items())
			res_print.sort(key=lambda x: x[0])
			for key, value in res_print:
				if isinstance(value, float):
					print("\t%s:\t%f" % (key, value))
					f.write("%s:\t%f\n" % (key, value))
			f.write('\n')
			for i in range(len(res['resp'])):
				f.write("resp:\t%s\n" % " ".join(res['resp'][i]))
				f.write("gen:\t%s\n\n" % " ".join(res['gen'][i]))

		print("result output to %s." % test_file)
		return {key: val for key, val in res.items() if type(val) in [bytes, int, float]}
