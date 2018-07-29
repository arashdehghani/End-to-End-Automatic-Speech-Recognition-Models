# ------------------------
# -    DeepSpeech2.py    - 
# ------------------------
# - Author:  Tao, Tu
# - Date:    2018/7/30
# - Description:
#      Deep speech 2 model.
#
# -----------------------

import tensorflow as tf
import numpy as np
import time
import math
import os

class DeepSpeech2:
    def __init__(self, sess, hps):
        self.num_features = hps.num_features
        self.num_classes = hps.num_classes
        self.num_hidden = hps.num_hidden
        self.num_rnn_layers = hps.num_rnn_layers
        self.keep_prob = 1 - hps.drop_prob
        self.lr = hps.lr
        self.grad_clip = hps.grad_clip
        self.batch_size = hps.batch_size
        self.sess = sess
        pass
    
    def build_graph(self, is_training=False):
        with tf.variable_scope('Input'):
            # Shape = [batch_size, time_step, num_features]
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.num_features])
            # Sparse representation is required for ctc_loss
            self.targets = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
            shape = tf.shape(self.inputs)
            batch_size, time_steps, num_features = shape[0], shape[1], shape[2]
            # Shape = [batch_size, num_features, time_step, 1]
            self.Xrs = tf.transpose(self.inputs, [0, 2, 1])
            self.Xrs = tf.expand_dims(self.Xrs, axis=-1)
            # self.Xrs = tf.reshape(self.inputs, shape=[batch_size, num_features, time_steps, 1])
            
        with tf.variable_scope('Convolution_layer'):
            # Shape = [height, width, in_channel, out_channel]
            filter_1 = tf.get_variable('filter_1', shape=(41, 11, 1, 32), 
                                       initializer=tf.truncated_normal_initializer(stddev=0.02), 
                                       dtype=tf.float32)
            stride_1 = [1, 2, 1, 1]
            filter_2 = tf.get_variable('filter_2', shape=(21, 11, 32, 32), 
                                       initializer=tf.truncated_normal_initializer(stddev=0.02), 
                                       dtype=tf.float32)
            stride_2 = [1, 2, 1, 1]
            filter_3 = tf.get_variable('filter_3', shape=(21, 11, 32, 96), 
                                       initializer=tf.truncated_normal_initializer(stddev=0.02), 
                                       dtype=tf.float32)
            stride_3 = [1, 2, 1, 1]
            
            layer_1 = tf.nn.conv2d(self.Xrs, filter_1, stride_1, padding='SAME')
            layer_1 = tf.nn.leaky_relu(layer_1)
            layer_1 = tf.layers.batch_normalization(layer_1, training=is_training)
            layer_1 = tf.contrib.layers.dropout(layer_1, keep_prob=self.keep_prob, is_training=is_training)
            
           
            layer_2 = tf.nn.conv2d(layer_1, filter_2, stride_2, padding='SAME')
            layer_2 = tf.nn.leaky_relu(layer_2) 
            layer_2 = tf.layers.batch_normalization(layer_2, training=is_training)
            layer_2 = tf.contrib.layers.dropout(layer_2, keep_prob=self.keep_prob, is_training=is_training)
            # The shape of layer_3's output is [batch_size, height, width, channels]
            # layer_3 = tf.nn.conv2d(layer_2, filter_3, stride_3, padding='SAME')
            # layer_3 = tf.layers.batch_normalization(layer_3, training=is_training)
            # layer_3 = tf.contrib.layers.dropout(layer_3, keep_prob=self.keep_prob, is_training=is_training)
            
        with tf.variable_scope('Recurrent_layer') as scope:
            seq_len_shrinked = tf.ceil(
                tf.div(tf.to_float(self.seq_len), stride_1[2] * stride_2[2])) #* stride_3[2]))
            seq_len_shrinked = tf.to_int32(seq_len_shrinked)
            num_features_shrinked = math.ceil(float(self.num_features) / (stride_1[1] * stride_2[1])) #* stride_3[1]))
            # Shape = [batch_size, time_steps_shrinked, num_features_shrinked, channel]
            rnn_input = tf.transpose(layer_2, (0, 2, 1, 3))
            # Shape = [batch_size, time_steps_shrinked, channel * num_features_shrinked]
            # If you want to try 3 conv. layers, change 32 to 96.
            rnn_input = tf.reshape(rnn_input, shape=[batch_size, -1, 32 * num_features_shrinked])
            for i in range(self.num_rnn_layers):
                cell_fw = tf.contrib.rnn.LSTMCell(self.num_hidden) 
                cell_bw = tf.contrib.rnn.LSTMCell(self.num_hidden) 
                rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=rnn_input, 
                    sequence_length=seq_len_shrinked, 
                    dtype=tf.float32,
                    time_major=False,
                    scope='RNN-%d' % i
                )
                o_rnn_fw, o_rnn_bw = rnn_output
                o_rnn = o_rnn_fw + o_rnn_bw
                o_rnn = tf.layers.batch_normalization(o_rnn, training=is_training)
                o_rnn = tf.contrib.layers.dropout(o_rnn, keep_prob=self.keep_prob, is_training=is_training)
                # output with shape [batch_size, time_steps_shrinked, num_hidden]
                rnn_input = o_rnn
                
        with tf.variable_scope('Projection_layer'):
            flatten = tf.reshape(o_rnn, shape=[-1, self.num_hidden])
            W_proj = tf.get_variable(name='W_proj', shape=([self.num_hidden, self.num_classes]),
                                     initializer=tf.truncated_normal_initializer(stddev=0.02), 
                                     dtype=tf.float32)
            b_proj = tf.get_variable(name='b_proj', shape=([self.num_classes]),
                                     initializer=tf.constant_initializer(value=0.0), 
                                     dtype=tf.float32)
            
            logits = tf.matmul(flatten, W_proj) + b_proj
            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_size, -1, self.num_classes])
            # Time major
            self.logits = tf.transpose(logits, (1, 0, 2))
            
        with tf.variable_scope('Loss'):
            self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len)
            self.cost = tf.reduce_mean(self.loss)

        with tf.variable_scope('Prediction'):
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
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.var_trainable = tf.trainable_variables()
            # Gradient clipping
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.cost, self.var_trainable), self.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-3)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.opt = optimizer.apply_gradients(
                    zip(grads, self.var_trainable), global_step=self.global_step)

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
                print(log.format(ep, batch, num_batch_per_epoch, batch_cost, batch_per, time.time() - start), flush=True)
        train_per /= num_train_example
        print('[!] epoch {}: train_per = {:.3f}'.format(ep, train_per), flush=True)
        self.save(ckpt_dir=ckpt_dir, idx=ep)
        
    def test(self, data):
        start = time.time()
        total_per = 0
        num_batch = data.num_samples // self.batch_size
        for b in range(num_batch):
            X, y = data[b]
            inputs_pad, seq_len = data.padding(X)
            targets_sparse = data.to_sparse_tuple(y, dtype=np.int64)
            #targets = y
            feed = {self.inputs: inputs_pad,
                    self.seq_len: seq_len, 
                    self.targets: targets_sparse}
            per, pred = self.sess.run([self.per, self.pred], feed_dict=feed)
            total_per += per * self.batch_size
            #log = "{}: test_per = {:.3f}, time = {:.3f}"
            #print(log.format(b, per, time.time() - start), flush=True)
        print('[*] testing PER: %f, time: %.3f' % (total_per/(num_batch*self.batch_size), time.time() - start), flush=True)
        
    
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
                print('[Restore] load latest check point')
        return exist_model
    
