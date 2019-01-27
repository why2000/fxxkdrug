# -*- coding: utf-8 -*-
# Author: Tree Wu, 2018.10, UCLA Extension

import tensorflow as tf
import collections
import numpy as np
import re
import os


# Data processor
class Process(object):
    def __init__(self, data_path):
        self._data_size = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(self._data_size)
        self._data_path = data_path
        self._poems_vec = []
        self._words_dict = {}

    def data_process(self):
        poems = []
        with open(self._data_path, 'r') as data_file:
            data_gen = data_file.readlines()
            for line in data_gen:
                title = line.strip('\n').strip(' ').replace(' ', '').split(':')
                poem = ''.join(title[1:])
                title = title[0]
                if '_' in poem or '(' in poem or '（' in poem or '《' in poem or '[' in poem or len(poem) < 5 or len(poem) > 79:
                    continue
                poem = '[' + poem + ']'
                poems.append(poem)
        words_poems = []
        for poem in poems:
            for word in poem:
                words_poems.append(word)
        # print(poems[0])
        poems = sorted(poems, key=lambda x: len(x))
        words_list, _ = zip(*sorted(collections.Counter(words_poems).items(), key=lambda x: x[-1]))
        words_list = words_list + (' ',)
        words_dict = dict(zip(words_list, range(len(words_list))))
        self._words_dict = words_dict
        poems_vec = [list(map(lambda x: words_dict[x], poem)) for poem in poems]
        self._poems_vec = poems_vec
        self._data_size = len(poems_vec)
        self._data_index = np.arange(self._data_size)
        # print(len(poems))
        return words_list, poems_vec, words_dict

    # ive a batch of data when called
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start + batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed + 1
            self._index_in_epoch = batch_size
            full_batch_features, full_batch_labels = self._get_batch_(0, batch_size)
            return full_batch_features, full_batch_labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_batch_features, full_batch_labels = self._get_batch_(start, end)
            if self._index_in_epoch == self._data_size:
                self._index_in_epoch = 0
                self._epochs_completed = self._epochs_completed + 1
                np.random.shuffle(self._data_index)
            return full_batch_features, full_batch_labels

    # Get a single batch
    def _get_batch_(self, start, end):
        batches = []
        for i in range(start, end):
            # print(len(self._poems_vec))
            batches.append(self._poems_vec[self._data_index[i]])

        length = max(map(len, batches))

        x_data = np.full((end - start, length), self._words_dict[' '], np.int32)
        for row in range(end - start):
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        return x_data, y_data


# RNN Model
class PoemRNN(object):
    def __init__(self, args, action='train'):
        if action == 'train':
            self.batch_size = args['batch_size']
        else:
            self.batch_size = 1
        self.data_processor = Process(args['data_path'])
        self.words_list, self.poems_vec, self.words_dict = self.data_processor.data_process()
        self.num_batches = len(self.poems_vec) // self.batch_size
        self.num_epochs = args['num_epochs']
        self.learning_rate = 0.002
        self.model_name = args['model_name']
        self.num_hidden_feature = args['num_hidden_feature']
        self.num_hidden_layers = args['num_hidden_layers']
        self.grad_clip = args['grad_clip']
        self.device = args['device']
        self.total_loss = 0.0
        self.epoch_loss = 0.0
        self.batch_loss = 0.0
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.loss = None
        self.optimizer = None
        self.cell = None
        self.logits = None
        self.input_data = None
        self.output_data = None
        self.writer = None
        self.sess = None
        self.cur_softmax_probs = None
        self.initial_state = None
        self.last_state = None
        self.epoch_loss_summary = None
        self.batch_loss_summary = None
        tf.reset_default_graph()
        self.build_placeholder()
        self.build_model()
        self.build_loss()
        self.build_train()
        self.build_graph()
        if action == 'train':
            self.train(pred=False)
        elif action == 'pred':
            self.pred('深度网络', 5)

    # set data shape
    def build_placeholder(self):
        with tf.name_scope('visible_data'):
            self.input_data = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None], name='input')
            self.output_data = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None], name='output')

    # set rnn model
    def build_model(self):
        with tf.name_scope('model'):
            cell_obj = None
            if self.model_name == 'RNN':
                cell_obj = tf.nn.rnn_cell.BasicRNNCell
            elif self.model_name == 'LSTM':
                cell_obj = tf.nn.rnn_cell.BasicLSTMCell
            self.cell = tf.nn.rnn_cell.MultiRNNCell([cell_obj(self.num_hidden_feature,name='basic_cell')]*self.num_hidden_layers)
            self.initial_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            w = tf.Variable(tf.truncated_normal([self.num_hidden_feature, len(self.words_list)], stddev=0.1), name='softmax_w')
            b = tf.Variable(tf.zeros(len(self.words_list)), name='softmax_b')
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [len(self.words_list), self.num_hidden_feature])
                cur_input = tf.nn.embedding_lookup(embedding, self.input_data)
            cur_output, self.last_state = tf.nn.dynamic_rnn(self.cell, cur_input, initial_state=self.initial_state)
            self.logits = tf.matmul(tf.reshape(cur_output, [-1, self.num_hidden_feature]), w) + b
            self.cur_softmax_probs = tf.nn.softmax(logits=self.logits)

    # set loss
    def build_loss(self):
        with tf.name_scope('loss'):
            y_sqeezed = tf.reshape(self.output_data, [-1])
            self.loss = tf.reduce_mean(tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits], [y_sqeezed], [tf.ones_like(y_sqeezed, dtype=tf.float32)], len(self.words_list)))
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_sqeezed), name='loss')

    # set optimizer
    def build_train(self):
        with tf.name_scope('optimizer'):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.optimizer = optimizer.apply_gradients(zip(grads, tvars))

    #load saved model
    def _model_loader(self, saver, checkpoint_path):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            print('resume from checkpoint ', latest_checkpoint)
            saver.restore(self.sess, latest_checkpoint)
            return int(latest_checkpoint[latest_checkpoint.rindex('-') + 1]) + 1
        else:
            print('building new model...')
            self.sess.run(tf.global_variables_initializer())
            return 0

    # for tensorboard
    def build_graph(self):
        with tf.name_scope('summaries'):
            self.epoch_loss_summary = tf.summary.scalar('epoch_loss', self.epoch_loss)
            self.batch_loss_summary = tf.summary.scalar('batch_loss', self.batch_loss)
        self.writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

    # training
    def train(self, pred=True):
        self.sess = tf.Session(config=self.sess_config)
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver(tf.all_variables())
        cur_epoch = self._model_loader(saver, './model/')
        for epoch in range(cur_epoch, cur_epoch + self.num_epochs):
            self.epoch_loss = 0.0
            for batch in range(self.num_batches):
                x, y = self.data_processor.next_batch(self.batch_size)
                # print(x)
                feed_dictionary = {
                    self.input_data: x,
                    self.output_data: y
                }
                self.batch_loss, _, _, batch_summary, epoch_summary= self.sess.run([self.loss, self.last_state, self.optimizer, self.batch_loss_summary, self.epoch_loss_summary], feed_dict=feed_dictionary)
                self.writer.add_summary(batch_summary, batch)
                self.epoch_loss = self.epoch_loss + self.batch_loss
            self.epoch_loss = self.epoch_loss / self.num_batches
            self.writer.add_summary(epoch_summary, epoch)
            saver.save(self.sess, 'model/poem.module', global_step=epoch)
            self.total_loss = self.total_loss + self.epoch_loss / self.num_batches
            print('Epoch ', epoch, ' Loss is: ', self.epoch_loss / self.num_batches)
        self.total_loss = self.total_loss / self.num_epochs
        print('Train Finished, Total Loss is: ', self.total_loss)
        self.sess.close()
        if pred:
            self.pred('深度网络', 5)

    # making prediction
    def pred(self, heads, type):
        if type != 5 and type != 7:
            print('The second para has to be 5 or 7!')
            return

        def to_word(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            sample = int(np.searchsorted(t, np.random.rand(1) * s))
            return self.words_list[sample]

        self.sess = tf.Session(config=self.sess_config)
        with tf.device(self.device):

            self.sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.all_variables())
            saver.restore(self.sess, 'model/poem.module-18')
            poems = []
            for head in heads:
                poem = ''
                flag = True
                while flag:
                    cur_stage = self.sess.run(self.cell.zero_state(1, tf.float32))
                    x = np.array([list(map(self.words_dict.get, '['))])
                    # print(x)
                    feed_dictionary = {
                        self.input_data:  x,
                        self.initial_state: cur_stage
                    }
                    [cur_probs, cur_stage] = self.sess.run([self.cur_softmax_probs, self.last_state], feed_dict=feed_dictionary)

                    sentence = head

                    x = np.zeros((1, 1))
                    x[0, 0] = self.words_dict[sentence]
                    [cur_probs, cur_stage] = self.sess.run([self.cur_softmax_probs, self.last_state], feed_dict=feed_dictionary)
                    word = to_word(cur_probs)
                    sentence += word
                    total_length = 0
                    now_length = 0
                    while word != '。':
                        x = np.zeros((1, 1))
                        x[0, 0] = self.words_dict[word]
                        [cur_probs, cur_stage] = self.sess.run([self.cur_softmax_probs, self.last_state], feed_dict=feed_dictionary)
                        word = to_word(cur_probs)
                        sentence += word
                        now_length += 1
                        if (now_length % (type * 2)) and (now_length % type == 0):
                            sentence += ','
                            now_length = 0
                            total_length += type
                        print(word)

                    if len(sentence) >= 2 + 2 * type:
                        sentence += u'\n'
                        poem += sentence
                        flag = False
                        print(sentence)
                poems.append(poem)
                print(poem)
            with open('./results/poems.txt', 'a+') as poems_file:
                for item in poems:
                    poems_file.write(item+'\n')
            self.sess.close()
            return poems


def run():
    args = {
        'batch_size': 64,
        'data_path': './data/poetry.txt',
        'num_epochs': 20,
        'learning_rate': 0.002,
        'model_name': 'LSTM',
        'num_hidden_feature': 128,
        'num_hidden_layers': 2,
        'grad_clip': 5,
        'device': '/gpu:0'
    }
    RNNmodel = PoemRNN(args, action='pred')


if __name__ == '__main__':
    run()