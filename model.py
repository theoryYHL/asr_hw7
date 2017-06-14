from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, GRUCell, LSTMCell, MultiRNNCell, DropoutWrapper, LayerNormBasicLSTMCell

######################################################
# 마지막 softmax layer에 사용
# 업로드 해주신 실습 자료 그대로 사용
######################################################
class Dense(object):

    def __init__(self, input_dim, output_dim, name='',
                 w_initializer=tf.random_normal_initializer(0.0, 0.02),
                 b_initializer=tf.zeros_initializer(),
                 use_bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        scope_name = name + '_Dense'
        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [input_dim, output_dim], tf.float32, w_initializer)
            self.b = tf.get_variable('b', [output_dim], tf.float32, b_initializer)

    def __call__(self, input_):
        """
        input_:     (batch_size, input_dim)
        """
        if self.use_bias:
            return tf.matmul(input_, self.W) + self.b
        else:
            return tf.matmul(input_, self.W)

    def get_params(self):
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]

######################################################
# 두 층의 lstm layer를 위해 사용
######################################################
class MODEL(object):

    def __init__(self, charsize, frame_length, batch_size, num_layer=2,
                 hidden_dim=500, keep_prob=0.5, name='MODEL'):
        self.charsize = charsize
        self.frame_length = frame_length
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.name = name

        Cell = LayerNormBasicLSTMCell
        cells_fw = []
        cells_bw = []
        with tf.variable_scope(self.name):
            for _ in range(num_layer):
                cell_fw = Cell(hidden_dim,dropout_keep_prob=keep_prob)
                cell_bw = Cell(hidden_dim,dropout_keep_prob=keep_prob)
                cells_fw.append(cell_fw)
                cells_bw.append(cell_bw)
            self.cells_fw = cells_fw
            self.cells_bw = cells_bw
            self.multicell_fw = MultiRNNCell(cells_fw)
            self.multicell_bw = MultiRNNCell(cells_bw)
            self.multicell_fw_zero = self.multicell_fw.zero_state(self.batch_size, tf.float32)
            self.multicell_bw_zero = self.multicell_bw.zero_state(self.batch_size, tf.float32)

            self.dense = Dense(hidden_dim, charsize, 'dense')
        self.params = None

    def __call__(self, input_, indices_, values_, shape_, seq_len_=None, is_train=False, reuse_=False):
        """
        주의
        input_:     (batch_size, frame_length, data_dim)
        h1:         (batch_size, sequence_length, hidden_dim)
        h2:         (batch_size * sequence_length, hidden_dim)
        h3:         (batch_size * sequence_length, phoneme)
        h3:         (batch_size , sequence_length, phoneme)
        output:     (batch_size , sequence_length, phoneme)
        """

        # tf.nn.batch_normalization()
        #
        # tf.sequence_mask()
        # training 단계면 dropout을 사용함
        with tf.variable_scope(self.name, reuse=reuse_):
            h1, s1 = tf.nn.bidirectional_dynamic_rnn(
                self.multicell_fw, self.multicell_bw,input_, seq_len_,dtype=tf.float32)
        h2 = tf.reshape(h1, [-1, self.hidden_dim])
        h3 = self.dense(h2)
        # output을 one_hot과 비교하기 위한
        h4 = tf.reshape(h3, [self.batch_size, -1, self.charsize])
        softmax = tf.nn.softmax(h4)
        output = tf.transpose(softmax,perm=[1,0,2])

        labels = tf.SparseTensor(indices_,values_,shape_)
        loss = tf.nn.ctc_loss(labels, output,seq_len_)
        greedy = tf.nn.ctc_greedy_decoder(output,seq_len_)
        beam = tf.nn.ctc_beam_search_decoder(output,seq_len_)
        """
        # Compute cross entropy for each frame
        cross_entropy = target_y * tf.log(output)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        # mask = tf.sequence_mask(seq_len_, dtype=tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(target_y), reduction_indices=2))
        cross_entropy *= mask

        # Average over actual sequence lengths
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)

        loss = [] # tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_y, logits=h4))
        self.params = tf.trainable_variables()
        # tf.contrib.rnn generates parameters after dynamic_rnn call.
        # you should first run dummy data or first batch to get all parameters.
        # or, you can make your own lstm layer.

        return h4, s1, cross_entropy
        """

        self.params = tf.trainable_variables()

        return loss, labels ,beam, output

    # zero_state 출력 함수
    def zero_state(self):
        return self.stack.zero_state(self.batch_size, tf.float32)


    # 계산된 parameter를 저장하는 함수. train에 사용
    def save_params(self, session):
        param_dir = './params_'
        for pp in self.params:
            value = session.run(pp)
            name_string = pp.name.split('/')
            path = param_dir + '/'.join(name_string[:-1]) + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + name_string[-1][:-2] + '.npy', value)  # remove :0


    # 저장된 파라미터를 불러오는 함수. evaluation에 사용
    def load_params(self, session):
        param_dir = './params_'
        # vars = {v.name: v for v in tf.trainable_variables()}
        for pp in self.params:
            name_string = pp.name.split('/')
            path = param_dir + '/'.join(name_string[:-1]) + '/'
            value = np.load(path + name_string[-1][:-2] + '.npy')
            session.run(pp.assign(value))

