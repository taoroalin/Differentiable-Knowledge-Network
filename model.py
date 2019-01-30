import numpy as np
import random
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
import layers

def missing_link_generator(batch_size, nodes, dimensions):
    while True:
        yield missing_link(batch_size, nodes, dimensions)

def missing_link(num_samples, num_nodes, dimensions):
    connections = num_nodes*(num_nodes-1) - 1
    samples = np.zeros((num_samples, connections, dimensions*2))
    targets = np.zeros((num_samples, dimensions*2))
    for sample in range(num_samples):
        skip=random.randint(0, num_nodes-1)
        nodes = [np.random.uniform(-1, 1, dimensions) for _ in range(num_nodes)]
        i=0
        skipped=False
        for comb in itertools.combinations(nodes, 2):
            if i!=skip and not skipped:
                samples[sample, i] = np.concatenate(comb)
                i+=1
            else:
                targets[sample]=np.concatenate(comb)
                skip=True
    return samples, targets

def attention(data, state, function):
    dataset = torch.cat([data, state.expand(data.size()[0], state.size()[0])], 1)
    weightings = function(dataset)
    weightings = f.softmax(torch.tanh(weightings), dim=0)
    return torch.mm(weightings.t(), data)

def subspace_attention(data, state, function, assemble):
    dataset = torch.cat([data, state.expand(data.size()[0], state.size()[0])], 1)
    weightings = function(dataset)
    weightings = f.softmax(torch.tanh(weightings), dim=0)
    return torch.mm(weightings.t(), assemble(dataset))

def attention_MISD(data, state, function):
    dataset = torch.cat([data, state.expand(data.size()[0], state.size()[0])], 1)
    weightings = function(dataset)
    weightings = f.softmax(torch.tanh(weightings), dim=0)
    return torch.mm(weightings.t(), data)

class Synthesizer(nn.Module):

    def __init__(self, dimensions, heads):
        super(Synthesizer, self).__init__()
        self.dimensions = dimensions
        self.heads = heads

        self.prioritize = nn.Linear(2*dimensions, 1)
        self.assemble = nn.Sequential(nn.ReLU(),nn.Linear(2*dimensions, dimensions))

    def forward(self, old_state, child_states):
        x = subspace_attention(child_states, old_state, self.prioritize, self.assemble).squeeze(0)
        return x

class Explorer(nn.Module):

    def __init__(self, dimensions, heads):
        super(Explorer, self).__init__()
        self.dimensions = dimensions
        self.heads = heads

        self.query = nn.Linear(2*dimensions, heads)
        self.read = nn.Sequential(nn.Linear(2*dimensions, dimensions), nn.ReLU())
        self.prune = nn.Linear(2*dimensions, 1)

    def forward(self, state, memory):
        active_memory = attention_MISD(memory, state, self.query)
        active_memory = torch.cat([active_memory, state.expand(active_memory.size()[0], -1)], -1)
        new_states = self.read(active_memory).view(self.heads, self.dimensions)
        priorities = attention(new_states, state, self.prune)
        return new_states, priorities

class RAMN(nn.Module):

    def __init__(self, dimensions, heads):
        super(RAMN, self).__init__()
        self.dimensions = dimensions
        self.heads = heads
        self.synth = Synthesizer(dimensions, heads)
        self.expl = Explorer(dimensions, heads)

        self.out = nn.Linear(dimensions, dimensions)

    def forward(self, memories, initial_states, depth):
        memories = torch.split(memories, 1, 0)
        initial_states = torch.split(initial_states, 1, 0)
        out = []
        for memory, initial_state in zip(memories, initial_states):
            out.append(self.out(self.query(memory.squeeze(0), initial_state.squeeze(0), depth)))
        return torch.stack(out, 0)
    
    def query(self, memory, initial_state, depth):
        child_states, priorities = self.expl(initial_state, memory)
        children = []
        if depth > 0:
            for head in torch.split(child_states, 1, dim=-2):
                children.append(self.query(memory, head.squeeze(0), depth-1))
            children = torch.stack(children, 0)
        else:
            children = child_states
        fstate = self.synth(initial_state, children)
        return fstate

def test_missing_link():
    dimensions = 12
    nodes = 4
    heads = 2
    depth = 4
    cuda = True

    decoder = RAMN(dimensions, heads)
    if cuda:
        decoder.cuda()
    optimizer = torch.optim.Adam(decoder.parameters())
    loss_fn = nn.MSELoss()

    batches = 100
    batch_size = 1000

    for i, (x, y)in enumerate(missing_link_generator(batch_size, nodes, dimensions//2)):
        torch_x = torch.from_numpy(x).float()
        torch_y = torch.from_numpy(y).float()
        initial_state = torch.zeros(batch_size, dimensions).float()
        if cuda:
            torch_x = torch_x.cuda()
            torch_y = torch_y.cuda()
            initial_state = initial_state.cuda()
        pred = decoder(torch_x, initial_state, depth)
        loss = loss_fn(pred, torch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outloss = float(loss.detach().cpu().numpy())
        print("Loss: {0:.5} Batch: {1}".format(outloss, i))

test_missing_link()