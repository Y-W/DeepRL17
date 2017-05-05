from __future__ import division

import numpy as np

def safe_div(a, b):
    return a / np.maximum(b, 1.0)

class BinBatchLearner:
    def __init__(self, bin_fn, bin_shape, output_shape):
        self.bin_shape = bin_shape
        self.output_shape = output_shape
        self.bin_fn = bin_fn
        self.bin_sum = np.zeros(self.bin_shape + self.output_shape, dtype=np.float_)
        self.bin_cnt = np.zeros(self.bin_shape + self.output_shape, dtype=np.int_)
    
    def reset(self):
        self.bin_cnt[...] = 0
        self.bin_sum[...] = 0.0

    def add_batch(self, X, A, Y):
        self.reset()
        assert X.shape[0] == Y.shape[0] and X.shape[0] == A.shape[0]
        for k in xrange(X.shape[0]):
            bin_idx = self.bin_fn(X[i])
            self.bin_sum[bin_idx][A[k]] += Y[k]
            self.bin_cnt[bin_idx][A[k]] += 1
    
    def _eval(self, bin_idx):
        return safe_div(self.bin_sum[bin_idx], self.bin_cnt[bin_idx])

    def eval(self, x):
        return self._eval(self.bin_fn(x))

    def eval_batch(self, X):
        result = np.zeros((X.shape[0],) + self.output_shape, dtype=np.float_)
        for k in xrange(X.shape[0]):
            result[k] = self.eval(X[i])
        return result
