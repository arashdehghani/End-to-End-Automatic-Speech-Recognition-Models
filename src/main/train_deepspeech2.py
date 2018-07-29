# --------------------------------
# -     train_deepspeech2.py     - 
# --------------------------------
# - Author:  Tao, Tu
# - Date:    2018/7/30
# - Description:
#      Train deep speech 2 model.
#
# --------------------------------

import time
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]='0'
curr_script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf
from utils.timit_dataset import TimitDataset, load_data
from utils.timit_preprocess import idx2phn, phonemes
from model.DeepSpeech2 import DeepSpeech2


# hyperparameter
hps_list = {
    'num_epochs': 200,
    'lr': 5e-3,
    'grad_clip': 5, 
    'num_hidden': 128,
    'num_features': 39,
    'num_classes': 61+1,
    'num_rnn_layers': 2,
    'batch_size': 32,
#     'max_time_step': 778,
    'drop_prob': 0.2
}

# Can be accessed as 'hps.attr'
def hparas(hps_list):
    class Hparas(object):
        pass
    hps = Hparas()
    for hp in hps_list:
        setattr(hps, hp, hps_list[hp])
    return hps



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 train_deepspeech2.py <timit_mfcc_path>')
        sys.exit(1)

    # Create a hyperparater object
    hps = hparas(hps_list)

    feat_path = sys.argv[1]
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data(feat_path)

    data_tr = TimitDataset(X_tr, y_tr, batch_size=hps.batch_size)
    # data_val = TimitDataset(X_val, y_val)
    data_te = TimitDataset(X_te, y_te, batch_size=hps.batch_size)

    # Used for .ipynb
    tf.reset_default_graph()

    sess = tf.Session()
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        model_tr = DeepSpeech2(sess, hps=hps)
        model_tr.build_graph(is_training=True)
        model_te = DeepSpeech2(sess, hps=hps)
        model_te.build_graph(is_training=False)
    
    # If you don't want to re-train a model, set it to False
    re_train = True 
    for ep in range(hps.num_epochs):
        data_tr.shuffle()
        model_tr.train(data_tr, ep, ckpt_dir='./ckpt_model', log=True, load_idx=None, re_train=re_train)
        re_train = False 
        # See testing PER after ep >= 5
        # if blah blah blah, test 
        if ep > 2:
            model_te.test(data_te)
        
    sess.close()
