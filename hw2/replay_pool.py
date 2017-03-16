from collections import deque
import random

class ReplayPool:
    def __init__(self, expected_size, age_limit):
        assert expected_size <= age_limit
        self.expected_size = expected_size
        self.age_limit = age_limit
        self.buffer = deque()
        self.timestamp = 0
        self.prob = float(self.expected_size) / float(self.age_limit)
    
    def add(self, obj):
        self.timestamp += 1
        while len(self.buffer) > 0 and self.buffer[0][0] <= self.timestamp - self.age_limit:
            self.buffer.popleft()
        if random.random() < self.prob:
            self.buffer.append((self.timestamp, obj))

    def extend(self, objs):
        for obj in objs:
            self.add(obj)
    
    def sample(self, batch_size):
        return [self.buffer[random.randrange(len(self.buffer))][1] for _ in xrange(batch_size)]

class NoPool:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque()
    
    def add(self, obj):
        while len(self.buffer) >= self.max_size:
            self.buffer.popleft()
        self.buffer.append(obj)

    def extend(self, objs):
        for obj in objs:
            self.add(obj)
    
    def sample(self, batch_size):
        return [self.buffer[i] for i in xrange(len(self.buffer) - batch_size, len(self.buffer))]
