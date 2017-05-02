import numpy as np

class BinBatchLearner:
    def __init__(self, preprosessor, bin_shape):
        self.bin_shape = bin_shape
        self.table = np.zeros(self.bin_shape, dtype=np.float_)
        self.preprosessor = preprosessor
    
    def train_batch(self, X, Y):
        self.table[:] = 0.0
        tmp_cnt = np.zeros(self.bin_shape, dtype=np.int_)
        for k, x in enumerate(self.preprosessor(X)):
            self.table[x] += Y[k]
            tmp_cnt[x] += 1
        
        tmp_avg = np.mean(self.table)
        self.table[:] += (tmp_cnt == 0) * tmp_avg
        tmp_cnt[:] += (tmp_cnt == 0)

        self.table[:] /= tmp_cnt
    
    def eval_batch(self, dataset):
        result = []
        for x in self.preprosessor(X):
            result.append(self.table[x])
        return result
