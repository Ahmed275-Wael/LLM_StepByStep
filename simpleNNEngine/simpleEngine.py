# This is a simple NN engine "We are literally applying the same procedures stated by the math behind it"
# To revise some of the math we need to achieve a better understanding of the NN keep the following in mind
# Firstly : Each node (y) in the NN affects the endPoint or the result (z) and its effect is represented by the gradient
# of the result w.r.t this node (y) = dz/dy
# Secondly : Each node is a function contributing to the result "What we are trying to do is take inputs and try to tune our NN
# to get the desired output , but How can we do that if the inputs aren't changed, Simply : change the weights and bias affecting
# each node function, But here what is our point of reference [How Can we know if we should increase or decrease each weight]
# this can be simply achieved by knowing the gradient [The effect of a certain node on the outcome] using gradient descent,
# also, for better performance We try to calculate the difference between our result and the desired result and
# we try to decrease that loss"


# This is a very brief explanation on how we use NN in general to acheive a desired output given an input data, There are questions
# like why do we use them and why not only use linear regression or logistic regression , why do we need to embed them in whole network
# of neurons to achieve better responses, U should look this up as this is focused on the implementaion of a neural network engine
# just like pytorch minus ofcourse the better performance and also this engine works on rank 0 tensors [scalars] while Pytorch uses
# matrices which are much more efficient.

import math
import random
import numpy as np
import matplotlib.pyplot as plt


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __radd__(self, other):  # other + self
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out
# You should wonder why the grad is accumalated not just equal to actual local derivative * upcoming derivatives acc
# to chain rule "The reason is that one node could be affecting more than one output and  =  only means the value of its grad on to the output
# is overwritten each time" Also, u can verify this mathematically
    def backward(self):

        topo = []
        visited = set()
        # topological sorting is needed to determine the right order from which we would start the backtrack
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


