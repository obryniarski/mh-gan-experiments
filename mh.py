import numpy as np
import torch
from time import time
from tqdm import tqdm
import gc

from torch.utils.data import DataLoader


from utils import *
from model import *

def accept(proposal_prob, current_prob, disc):
    # print( (1/current_prob - 1) / (1/proposal_prob - 1))
    return np.fmin(1.0, (1/current_prob - 1) / (1/proposal_prob - 1))


def torch_accept(proposal_prob, current_prob):
    return torch.min(torch.ones(1).cuda(), (1/current_prob - 1) / (1/proposal_prob - 1))



def metropolis_hastings(gen, disc, K, initial, real_initial=True):
    print('um')
    changed = False
    x = initial
    for k in range(K):
        x_prime = gen(generate_noise(1, gen.module.nz))
        u = torch.rand(1, device=0)
        try:
            current_prob = disc(x).cpu().detach().numpy()[0][0]
            proposal_prob = disc(x_prime).cpu().detach().numpy()[0][0]
        except:
            current_prob = disc(x).cpu().detach().numpy()[0]
            proposal_prob = disc(x_prime).cpu().detach().numpy()[0]
        with np.errstate(divide='ignore'):
            a = accept(proposal_prob, current_prob, disc)
        if u <= a:
            changed = True
            x = x_prime

    if changed or not real_initial:
        return x
    # restarts += 1
    return metropolis_hastings(gen, disc, K, gen(generate_noise(1, gen.module.nz)), False)

def generate_mh_samples(num, gen, disc, K, initials):
    assert initials.shape[0] == num
    samples = []
    # restarts = 0

    for i in tqdm(range(num)):
        sample = metropolis_hastings(gen, disc, K, initials[i].view(-1, 2))
        samples.append(sample)
    # print('%i total restarts' % restarts)


    return torch.cat(samples, dim=0)


def equal(ar1, ar2):
    return torch.tensor([[torch.equal(ar1[i], ar2[i])] * 2 for i in range(len(ar1))]).cuda()

def vector_metropolis_hastings(gen, disc, K, initials, batch_size=100, real_initial=True):
    assert len(initials) == batch_size, 'wrong number of initial samples'
    x = initials
    for k in range(K):
        x_prime = gen(generate_noise(batch_size, gen.module.nz))
        u = torch.rand(batch_size, device=0)
        try:
            current_prob = disc(x).view(-1)
            proposal_prob = disc(x_prime).view(-1)
        except:
            print('HIIII')
            current_prob = disc(x)[0]
            proposal_prob = disc(x_prime)[0]
        with np.errstate(divide='ignore'):
            a = torch_accept(proposal_prob, current_prob)

        x = torch.where(torch.cat([u.view(-1, 1), u.view(-1, 1)], dim=1) <= torch.cat([a.view(-1, 1), a.view(-1, 1)], dim=1), x_prime, x)

    # run all the restarts together, from generated samples
    if real_initial:
        gen_initials = gen(generate_noise(batch_size, gen.module.nz))
        for_the_rejects = vector_metropolis_hastings(gen, disc, K, gen_initials, batch_size=len(gen_initials), real_initial=False)
        return torch.where(equal(x, initials), for_the_rejects, x)
    return x

def generate_vectorized_mh_samples(num, gen, disc, K, initials, batch_size=100):
    assert initials.shape[0] == num
    samples = []

    for i in tqdm(range(0, num, batch_size)):
        initials_batch = initials[i:i+batch_size]
        sample = vector_metropolis_hastings(gen, disc, K, initials_batch.view(-1, 2), batch_size)
        samples.append(sample)



    return torch.cat(samples, dim=0)
