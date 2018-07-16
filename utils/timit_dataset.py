# ------------------------
# -   timit_dataset.py   - 
# ------------------------
# - Author:  Tao, Tu
# - Date:    2018/7/16
# - Description:
#      For processing timit data conveniently.
#
# -----------------------


import pickle
import numpy as np
import random

def load_data(path):
    with open(path, 'rb') as handle:
        X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(handle)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

class TimitDataset:
    """For processing TIMIT data conveniently."""
    def __init__(self, X_all, y_all, batch_size=None):
        self.batch_size = batch_size
        self.num_samples = len(y_all)
        if batch_size is None:
            # Used for validation stage or when batch_size == 1
            self._X = X_all
            self._y = y_all
        else:
            self._X, self._y = self._do_batch(X_all, y_all)
            
    def __getitem__(self, idx):
        # Access a batch if batch_size is not None, else access a sample
        return self._X[idx], self._y[idx]

    def __len__(self):
        return len(self._y) 

    def shuffle(self, completely_shuffle=True):
        num_batch = len(self._X)
        if completely_shuffle and (self.batch_size is not None):
            # Shuffle all the samples
            X_all = []
            y_all = []
            for b in range(num_batch):
                for s in range(len(self._X[b])):
                    # b: batch, s: sample
                    X_all.append(self._X[b][s])
                    y_all.append(self._y[b][s])
            num_sample = len(y_all)
            idx = [i for i in random.sample(range(num_sample), num_sample)]
            X_all = [X_all[i] for i in idx]
            y_all = [y_all[i] for i in idx]
            self._X, self._y = self._do_batch(X_all, y_all)
        else:
            # Just shuffle the batches
            idx = [i for i in random.sample(range(num_batch), num_batch)]
            self._X = [self._X[i] for i in idx]
            self._y = [self._y[i] for i in idx]
        
    def _do_batch(self, X_all, y_all):
        # Return a list of batches
        num_sample = len(y_all)
        X_batched = []
        y_batched = []
        for batch in range(num_sample // self.batch_size):
            start = batch * self.batch_size
            end = (batch + 1) * self.batch_size
            if end >= num_sample:
                end = num_sample
            X_batched.append(X_all[start:end])
            y_batched.append(y_all[start:end])
        return X_batched, y_batched
    
    def to_sparse_tuple(self, sequences, dtype):
        """Turn sequences into sparse representations.
        [Args]
            sequences: a list whose element is a sequence with data type dtype
            dtype: as above

        [Returns]
            Sparse representation of (indices, values, shape)

        """
        indices = []
        values = []
        for n, seq in enumerate(sequences):
            indices.extend(zip([n]*len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), indices[:, 1].max() + 1], dtype=np.int64)
        return indices, values, shape
    
    def padding(self, sequences, max_length=None):
        """Pad the input sequences to max_length.
        [Args]
            sequences: a list whose element is a sequence of features

        [Returns]
            pad_x, original_lengths

        """
        lengths = np.asarray([len(seq) for seq in sequences], dtype=np.int64)
        if max_length is None:
            max_length = np.max(lengths)
        size = len(sequences)
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break
        x = (np.ones((size, max_length) + sample_shape) * 0).astype(np.float64)
        for idx, seq in enumerate(sequences):
            len_seq = len(seq)
            x[idx, :len_seq, :] = seq
        return x, lengths
    

