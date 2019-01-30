import torch
import torch.nn as nn
import numpy as np

def SubspaceSimilarityAccess(key, query_space, result_space, memory):
    k = key*query_space
    attention = nn.Softmax(torch.bmm(k, memory))
    raw = torch.sum(memory*attention.expand(memory.size()), 1)
    result = raw*result_space + (1-result_space)*key
    return result

def initMemory(dimensions):
    return torch.zeros(dimensions)

class Inferer(nn.Module):

    def __init__(self, dimensions, slots):
        super(Inferer, self).__init__()
        self.dimensions = dimensions
        self.slots = slots

        self.querior = nn.Linear(dimensions*2, dimensions)
        self.filterer = nn.Linear(dimensions*2, dimensions)
        self.synthesizer = nn.Linear(dimensions*(slots+1), dimensions)
        self.revealer = nn.Linear(dimensions, 1)
        
        self.heads = nn.Parameter(torch.empty(slots, dimensions).uniform_(-0.8, 0.8))
        self.objective = nn.Parameter(torch.empty(dimensions).uniform_(-0.8, 0.8))
        self.state = nn.Parameter(torch.zeros(dimensions))

    def forward(self, memory):
        qspace = self.querior(torch.cat([self.state.unsqueeze(0).expand(self.slots, self.dimensions), self.heads], 1))
        qspace = torch.tanh(qspace)
        rspace = self.querior(torch.cat([self.state.unsqueeze(0).expand(self.slots, self.dimensions), self.heads], 1))
        rspace = torch.tanh(rspace)
        self.heads = SubspaceSimilarityAccess(self.heads, qspace, rspace, memory)
        self.state = self.synthesizer(torch.cat([self.state, self.heads.flatten()]))
        reveal = torch.sigmoid(self.revealer(self.state))
        return self.state*reveal


class Encoder(nn.Module):

    def __init__(self, dimensions, input_size):
        super(Encoder, self).__init__()
        self.initial = nn.Sequential(nn.Linear(input_size, dimensions))
        #self.attention = nn.Linear(dimensions*2, dimensions)
        #self.filterer = nn.Linear(dimensions*2, dimensions)

    def forward(self, inpt, memory):
        k = self.initial(inpt)
        return torch.cat([memory, k], axis=0)


class Compressor(nn.Module):

    def __init__(self, dimensions):
        super(Compressor, self).init__()
    def forward(self, memory):
        pass