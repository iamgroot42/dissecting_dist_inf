import torch as ch
import torch.nn as nn


# fake relu function
class fakerelu(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# Fake-ReLU module wrapper
class FakeReluWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(FakeReluWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: ch.Tensor):
        return fakerelu.apply(input)


# identity function
class basic(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# identity function module wrapper
class BasicWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(BasicWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: ch.Tensor):
        return basic.apply(input)
