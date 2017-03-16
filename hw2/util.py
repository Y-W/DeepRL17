import numpy as np
from math import log, exp
from datetime import datetime

def list2NumpyBatches(l, batch_size):
    if len(l) == 0:
        return []
    shape = (batch_size,) + l[0].shape
    dtype = l[0].dtype
    p = 0
    result = []
    while p < len(l):
        delta = min(batch_size, len(l) - p)
        new_batch = np.zeros(shape, dtype=dtype)
        for i in xrange(delta):
            new_batch[i] = l[p + i]
        result.append(new_batch)
        p += delta
    return result

def numpyBatches2list(batches, original_length):
    result = []
    for b in batches:
        for i in xrange(b.shape[0]):
            result.append(b[i])
    return result[:original_length]

class Constant:
    def __init__(self, val):
        self.val = val
    def __call__(self):
        return self.val

class LinearDecay:
    def __init__(self, start_val, end_val, steps):
        self.start_val = float(start_val)
        self.end_val = float(end_val)
        self.steps = steps
        self.slope = float(end_val - start_val) / float(steps)
        self.n = 0
    def __call__(self):
        result = self.start_val + self.slope * self.n
        self.n += 1
        return result

class ExponentialDecay:
    def __init__(self, start_val, end_val, steps):
        self.start_val = float(start_val)
        self.end_val = float(end_val)
        self.steps = steps
        self.slope = log(self.end_val / self.start_val) / float(steps)
        self.n = 0
    def __call__(self):
        result = self.start_val * exp(self.slope * self.n)
        self.n += 1
        return result

def stats(l):
    n = len(l)
    l = np.array(l)
    u = np.mean(l)
    sigma = np.sqrt(np.sum((l - u) ** 2) * (1.0 / (n - 1.0)))
    return n, u, sigma, np.min(l), np.max(l)

def stats_str(st):
    return 'n=%i; mean=%f; std=%f; min=%f; max=%f' % st

def current_time():
    return '[' + str(datetime.now()) + '] '
