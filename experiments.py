import numpy as np
import random
import itertools
import math
import torch
import torch.nn as nn
import torch.utils.data as data
import layers

def missing_link_generator(batch_size, nodes, dimensions):
    while True:
        yield missing_link(batch_size, nodes, dimensions)

def missing_link(samples, nodes, dimensions):
    connections = nodes*(nodes-1) - 1
    samples = np.zeros((samples, connections, nodes*2))
    targets = np.zeros((samples, dimensions*2))
    for sample in range(samples):
        skip=random.randint(0, nodes-1)
        nodes = [np.random.uniform(-1, 1, dimensions) for _ in range(nodes)]
        i=0
        skipped=False
        for comb in itertools.combinations(nodes):
            if i!=skip and not skipped:
                connections[sample, i] = np.concatenate(comb)
                i+=1
            else:
                targets[sample]=np.concatenate(comb)
                skip=True
    return samples, targets

def missing_link_noisy(samples, junk_nodes, junk_edges, active_nodes, dimensions):
    csamples, targets = missing_link(samples, active_nodes, dimensions)
    samples = np.concatenate(csamples, csamples[:, :junk_edges], 1)
    for sample in range(samples):
        junk = [np.random.uniform(-1, 1, dimensions) for _ in range(junk_nodes)]
        for junk in range(junk_edges):
            samples[sample, junk, dimensions:] = random.choice(junk)
    return samples, targets

def train_missing_link():
    dimensions = 6
    nodes = 4
    heads = 2

    decoder = layers.Inferer(dimensions*2, heads)
    optimizer = torch.optim.Adam(decoder.parameters())
    loss_fn = nn.MSELoss()

    epochs = 10
    batch_size = 1000

    for i, (x, y)in enumerate(missing_link_generator(batch_size, nodes, dimensions)):
        cuda_x = torch.from_numpy(x).cuda()
        cuda_y = torch.from_numpy(y).cuda()
        pred = decoder(cuda_x)
        loss = loss_fn(pred, cuda_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outloss = float(loss.detach().cpu().numpy())
        print("Loss: {0:f5} Batch: {1}".format(outloss, i))
