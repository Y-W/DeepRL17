from __future__ import division

import numpy as np

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
        assert X.shape[0] == Y.shape[0] and X.shape[0] == A.shape[0]
        for k in xrange(X.shape[0]):
            bin_idx = self.bin_fn(X[k])
            self.bin_sum[bin_idx][A[k]] += Y[k]
            self.bin_cnt[bin_idx][A[k]] += 1
    
    def _eval(self, bin_idx, no_default=False):
        if not no_default:
            return np.where(self.bin_cnt[bin_idx] > 0, self.bin_sum[bin_idx] / self.bin_cnt[bin_idx], 0.0)
        else:
            return np.where(self.bin_cnt[bin_idx] > 0, self.bin_sum[bin_idx] / self.bin_cnt[bin_idx], np.nan)

    def eval(self, x, no_default=False):
        return self._eval(self.bin_fn(x), no_default=no_default)

    def eval_batch(self, X, no_default=False):
        result = np.zeros((X.shape[0],) + self.output_shape, dtype=np.float_)
        for k in xrange(X.shape[0]):
            result[k] = self.eval(X[k], no_default=no_default)
        return result
    
    def eval_batch_no_default(self, X):
        return self.eval_batch(X, no_default=True)
