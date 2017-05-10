import numpy as np

class BinIncLearner:
    def __init__(self, bin_fn, bin_shape, output_shape, alpha):
        self.bin_shape = bin_shape
        self.output_shape = output_shape
        self.bin_fn = bin_fn
        self.alpha = alpha
        self.bin_table = np.zeros(self.bin_shape + self.output_shape, dtype=np.float_)
    
    def reset(self):
        self.bin_table[...] = 0.0

    def update_batch(self, X, A, Y):
        assert X.shape[0] == Y.shape[0] and X.shape[0] == A.shape[0]
        for k in xrange(X.shape[0]):
            bin_idx = self.bin_fn(X[k])
            self.bin_table[bin_idx][A[k]] += self.alpha * (Y[k] - self.bin_table[bin_idx][A[k]])
    
    def _eval(self, bin_idx):
        return self.bin_table[bin_idx]

    def eval(self, x):
        return self._eval(self.bin_fn(x))

    def eval_batch(self, X):
        result = np.zeros((X.shape[0],) + self.output_shape, dtype=np.float_)
        for k in xrange(X.shape[0]):
            result[k] = self.eval(X[k])
        return result
