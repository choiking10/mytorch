import weakref
from queue import PriorityQueue
import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported.')
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0  # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        seen_set = set()
        funcs = PriorityQueue()
        funcs.put(self.creator)
        seen_set.add(self.creator)

        while funcs.qsize() != 0:
            f = funcs.get()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None and x.creator not in seen_set:
                    funcs.put(x.creator)
                    seen_set.add(x.creator)

    def zerograd(self):
        self.grad = None

    def __repr__(self):
        return f"data: {self.data} grad: {self.grad} creator: {self.creator} generation: {self.generation}"
