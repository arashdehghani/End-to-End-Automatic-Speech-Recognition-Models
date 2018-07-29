# ------------------------
# -       BiRNN.py       - 
# ------------------------
# - Author:  Tao, Tu
# - Date:    2018/7/16
# - Description:
#      Bidirectional recurrent neural network + ctc_loss.
#
# -----------------------

import numpy as np
import tensorflow as tf
from utils.timit_dataset import TimitDataset
from utils.timit_preprocess import idx2phn, phonemes
import time
import os


class BiRNN:
    """A bidrectional RNN for Automatic Speech Recognition."""
    def __init__(self, sess, is_training, hps):
        self.sess = sess
        self.lr = hps.lr
        self.grad_clip = hps.grad_clip
        self.num_hidden = hps.num_hidden
        self.num_features = hps.num_features
        self.num_classes = hps.num_classes
        self.num_cells = hps.num_cells
        self.num_layers = hps.num_layers
        self.batch_size = hps.batch_size
        # self.max_time_step = hps.max_time_step
        self.drop_prob = hps.drop_prob
        
        self.build_graph(is_training)
    
    
    def build_graph(self, is_training=False):
        with tf.variable_scope('Input'):
            self.inputs = tf.placeholder(tf.float32, [None, None, self.num_features])
            # Sparse representation is required for ctc_loss
            self.targets = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
            shape = tf.shape(self.inputs)
            batch_size, max_time_step = shape[0], shape[1]
            
        with tf.variable_scope('Rnn') as scope:
            cells_fw = []
            cells_bw = []
            for _ in range(self.num_cells):
                cell_fw = tf.contrib.rnn.LSTMCell(self.num_hidden)
                cell_bw = tf.contrib.rnn.LSTMCell(self.num_hidden)
                cells_fw.append(cell_fw)
                cells_bw.append(cell_bw)
            cells_fw = tf.contrib.rnn.MultiRNNCell(cells_fw)
            cells_bw = tf.contrib.rnn.MultiRNNCell(cells_bw)
            
            o_rnn, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells_fw,
                cell_bw=cells_bw,
                inputs=self.inputs, 
                sequence_length=self.seq_len, 
                dtype=tf.float32,
                time_major=False,
                scope=scope
            )
        
            o_rnn_fw, o_rnn_bw = o_rnn
            o_rnn = tf.concat([o_rnn_fw, o_rnn_bw], 2)
            o_rnn = tf.contrib.layers.dropout(o_rnn, keep_prob=1-self.drop_prob, is_training=is_training)
            
        with tf.variable_scope('FC_layers'):
            o_reshape = tf.reshape(o_rnn, [-1, 2*self.num_hidden])
            W_proj = tf.get_variable(name='W_proj', shape=([2*self.num_hidden, self.num_classes]),
                                     initializer=tf.truncated_normal_initializer(stddev=0.02), dtype=tf.float32)
            b_proj = tf.get_variable(name='b_proj', shape=([self.num_classes]),
                                     initializer=tf.constant_initializer(value=0.0), dtype=tf.float32)
            
            logits = tf.matmul(o_reshape, W_proj) + b_proj
            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_size, -1, self.num_classes])
            # Time major
            self.logits = tf.transpose(logits, (1, 0, 2))

        with tf.variable_scope('Loss'):
            self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len)
            self.cost = tf.reduce_mean(self.loss)

        with tf.variable_scope('Prediction'):
            # or ctc_greedy_decoder
            if is_training:
                self.decoded, log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len)
            else:
                self.decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len)
                
            # Inaccuracy: Phoneme Error Rate (PER)
            self.per = tf.reduce_mean(tf.edit_distance(
                tf.to_int32(self.decoded[0]), self.targets, normalize=True))
            # Prediction
            self.pred = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            
        with tf.variable_scope('Optimizer'):
            self.var_trainable = tf.trainable_variables()
            # Gradient clipping
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.cost, self.var_trainable), self.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-3)
            self.opt = optimizer.apply_gradients(zip(grads, self.var_trainable))

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=20)
    
    
    def train(self, data, ep, ckpt_dir='./ckpt_model', log=True, load_idx=None, re_train=False):
        num_train_example = data.num_samples
        num_batch_per_epoch = num_train_example // self.batch_size
        
        # train an epoch
        if not re_train:
            if load_idx:
                self.restore(ckpt_dir, idx=load_idx)
            else:
                self.restore(ckpt_dir)
        
        train_per = 0
        for batch in range(num_batch_per_epoch):
            start = time.time()
            batch_X, batch_y = data[batch]
            batch_train_inputs, batch_train_seq_len = data.padding(batch_X)
            batch_train_targets = data.to_sparse_tuple(batch_y, dtype=np.int64)
            feed = {self.inputs: batch_train_inputs,
                    self.targets: batch_train_targets,
                    self.seq_len: batch_train_seq_len}

            batch_cost, _ = self.sess.run([self.cost, self.opt], feed)
            batch_per = self.sess.run(self.per, feed_dict=feed)
            train_per += batch_per * self.batch_size
            if log and (batch % 1 == 0):
                log = "{}:{}/{}, train_cost = {:.3f}, train_per = {:.3f}, time = {:.3f}"
                print(log.format(ep, batch, num_batch_per_epoch, batch_cost, batch_per, time.time() - start))
        train_per /= num_train_example
        print('[!] epoch {}: train_per = {:.3f}'.format(ep, train_per))
        self.save(ckpt_dir=ckpt_dir, idx=ep)
        
        
    def test(self, data, num=5):
        start = time.time()
        X, y = data[:num]
        inputs_pad, seq_len = data.padding(X)
        targets_sparse = data.to_sparse_tuple(y, dtype=np.int64)
        targets = y
        feed = {self.inputs: inputs_pad,
                self.seq_len: seq_len, 
                self.targets: targets_sparse}
        per, pred = self.sess.run([self.per, self.pred], feed_dict=feed)
        print('[!] PER: %f, time: %.3f' % (per, time.time() - start))
        
        
    def save(self, ckpt_dir='./ckpt_model', idx=0):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.saver.save(self.sess, os.path.join(
            ckpt_dir, 'model-%d.ckpt' % idx))
    
    
    def restore(self, ckpt_dir='./ckpt_model', idx=None):
        exist_model = False
        if idx:
            self.saver.restore(self.sess, 
                               os.path.join(ckpt_dir, 'model-%d.ckpt' % idx))
            exist_model = True
        else:
            latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
            if latest_ckpt:
                self.saver.restore(self.sess, latest_ckpt)
                exist_model = True
        return exist_model
    
    def log(self):
        pass
    
