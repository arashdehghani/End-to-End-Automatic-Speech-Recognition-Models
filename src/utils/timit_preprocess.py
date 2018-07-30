# -----------------------
# - timit_preprocess.py -
# -----------------------
# - Author:  Tao, Tu
# - Date:    2018/7/16
# - Description:
#      Preprocessing Timit dataset into mfcc features.
#      Usage: python3 timit_preprocess.py <timit_directory> <output_fileName>
#
# ----------------------

import os
import pickle
from python_speech_features import base
import scipy.io.wavfile as wav
import random
import numpy as np
import time
import sys


# configs
NIST_extension = '.wav'
RIFF_extension = '_riff.wav'
PHN_extension = '.phn'

# validation set ratio
valid_ratio = 0.1

phonemes = [ 
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 
    'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 
    'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 
    'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 
    'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 
    'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 
    'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 
    'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 
    'v', 'w', 'y', 'z', 'zh'
]

# Turn phonemes into indices for training
# Use phonemes[phn2idx[phn]] to recover
phn2idx = {phn: idx for idx, phn in enumerate(phonemes)}

def idx2phn(idx):
    return phonemes[idx] if idx != -1 else '<BLANK>'

def extract_feature(wav_path):
    """Extract 39-dim mfcc feature."""
    fs, audio = wav.read(wav_path)
    mfcc    = base.mfcc(audio, fs, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, preemph=0.97, appendEnergy=True)
    mfcc_d  = base.delta(mfcc, N=2)
    mfcc_dd = base.delta(mfcc_d, N=2)
    feat = np.concatenate([mfcc, mfcc_d, mfcc_dd], axis=1)
    return feat

def compute_mean_std(X):
    """Compute the mean and standard deviation of the feature list X. Each element in the list is 
    of shape (time steps, feature dimensions). Note that the time steps of each element may differ 
    from others.
    """
    total = 0
    feat_dim = len(X[0][0])
    feat_sum = np.zeros(feat_dim)
    feat_square_sum = np.zeros(feat_dim)
    # calculate mean
    for xs in X:
        feat_sum += np.sum(xs, axis=0)
        total += len(xs)
    feat_mean = feat_sum / total
    # calculate std
    for xs in X:
        feat_square_sum += np.sum((xs - feat_mean)**2, axis=0)
    feat_std = (feat_square_sum / total)**(1/2)
    return feat_mean, feat_std

def standardize(X, dtype=np.float32):
    """Standardize X, i.e., (X - mean) / std."""
    mean, std = compute_mean_std(X)
    for i in range(len(X)):
        X[i] = ((X[i] - mean) / std).astype(dtype)
    return X

def preprocess(timit_path):
    """Preprocessing the timit dataset."""
    X = []
    y = []
    for root, dirs, files in os.walk(timit_path):
        for fname in files:
            if not fname.endswith(PHN_extension):
                continue

            phn_fname = os.path.join(root, fname)
            wav_fname = os.path.join(root, fname[:-4] + RIFF_extension)
            feat = extract_feature(wav_fname)
            with open(phn_fname, 'r') as f:
                label = [phn2idx[line.split()[-1]] for line in f.readlines()]
            X.append(feat)
            y.append(label) 
    return X, y



if __name__ == '__main__':
    start = time.time()
    if len(sys.argv) != 3:
        print('Usage: python3 timit_preprocess.py <timit_path> <output_fileName>')
        sys.exit(1)
    
    timit_path = sys.argv[1]
    output_file = sys.argv[2]
    print('Extracting mfcc features...')
    X_train_all, y_train_all = preprocess(os.path.join(timit_path, 'train'))
    X_test, y_test = preprocess(os.path.join(timit_path, 'test'))  
    
    print('Standardizing mfcc features...')
    X_train_std_all = standardize(X_train_all)
    X_test_std = standardize(X_test)
    
    print('Creating validation set...')
    num_train = len(y_train_all)
    val_idx = [i for i in random.sample(range(0, num_train), int(num_train*valid_ratio))]
    X_train_std = [X_train_std_all[i] for i in range(num_train) if i not in val_idx]
    X_valid_std = [X_train_std_all[i] for i in range(num_train) if i in val_idx]
    y_train = [y_train_all[i] for i in range(num_train) if i not in val_idx]
    y_valid = [y_train_all[i] for i in range(num_train) if i in val_idx]
    
    print('Dumping output files...')
    with open(os.path.join(output_file), 'wb') as handle:
        pickle.dump([X_train_std, y_train, 
                     X_valid_std, y_valid, 
                     X_test, y_test], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Total time: %.3f (sec)' % (time.time() - start))
    

