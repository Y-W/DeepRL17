from math import floor

class Discretization:
    def __init__(self, dim_range, dim_slots):
        self.dim_range = dim_range
        self.dim_slots = dim_slots
        self.ndim = len(dim_range)
        assert len(dim_slots) = self.ndim
        self.bin_size = tuple(float(self.dim_range[k][1] - self.dim_range[k][0]) / self.dim_slots[k] for k in xrange(self.ndim))

    def __call__(self, x):
        assert x.shape == (self.ndim,)
        assert all((x[k] >= self.dim_range[k][0]) and (x[k] <= self.dim_range[k][1]) for k in xrange(self.ndim))
        return tuple(min(int(floor(float(x[k] - self.dim_range[k][0]) / self.bin_size[k])), self.dim_slots[k] - 1) for k in xrange(self.ndim))
