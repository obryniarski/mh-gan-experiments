import numpy as np
import torch
from time import time

from utils import *
from model import *

def accept(proposal_prob, current_prob, disc):
    return np.fmin(1.0, (1/current_prob - 1) / (1/proposal_prob - 1))



def metropolis_hastings(gen, disc, K, initial, real_initial=True):
    changed = False
    x = initial
    for k in range(K):
        x_prime = gen(generate_noise(1, gen.module.nz))
        u = torch.rand(1, device=0)
        try:
            current_prob = disc(x).cpu().detach().numpy()[0][0]
            proposal_prob = disc(x_prime).cpu().detach().numpy()[0][0]
        except:
            current_prob = disc(x)[0]
            proposal_prob = disc(x_prime)[0]
        with np.errstate(divide='ignore'):
            a = accept(proposal_prob, current_prob, disc)
        if u <= a:
            changed = True
            x = x_prime

    if changed or not real_initial:
        return x
    print('spooky')
    return metropolis_hastings(gen, disc, K, gen(generate_noise(1, gen.module.nz)), False)

def generate_mh_samples(num, gen, disc, K, initials):
    assert initials.shape[0] == num
    samples = []
    total_time = 0
    while len(samples) < num:
        avg_time = total_time / len(samples) if total_time != 0 else 1
        seconds_remaining = int((num - len(samples)) * avg_time)
        print(len(samples), " - estimated time remaining: %i mins, %i seconds" % (seconds_remaining // 60, seconds_remaining % 60))
        start = time()
        sample = metropolis_hastings(gen, disc, K, initials[len(samples)].view(-1, 2))
        total_time += (time() - start)
        samples.append(sample)

    return torch.cat(samples, dim=0)
