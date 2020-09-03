# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Earlier versions of this file were written by Zhengdao Chen, used with permission.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from scipy.integrate import solve_ivp

from model import numerically_integrate


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


class SprChainHamilt(nn.Module):
    def __init__(self, k, m):
        super(SprChainHamilt, self).__init__()
        self.m = m
        self.k = k

    def forward(self, p_vec, q_vec):
        p, q = p_vec, q_vec
        kinetic = 0.5 * (p.pow(2) / self.m).sum(dim=1)
        q_diff = q[:, :-1] - q[:, 1:]
        potential = 0.5 * (self.k[1:-1] * q_diff.pow(2)).sum(dim=1) + 0.5 * (self.k[0] * q[:, 0].pow(2)) + 0.5 * (self.k[-1] * q[:, -1].pow(2))
        return kinetic + potential


class ThreeBodyHamilt(nn.Module):
    def __init__(self, masses):
        super(ThreeBodyHamilt, self).__init__()
        self.masses = masses
        self.G = 1

    def potential(self, q_vec):
        potential = torch.zeros(q_vec.size(0))
        q1 = q_vec[:, :2]
        q2 = q_vec[:, 2:4]
        q3 = q_vec[:, 4:]
        q_lst = [q1, q2, q3]
        for i in range(3):
            for j in range(i+1, 3):
                r_ij = ((q_lst[i] - q_lst[j])**2).sum(1)**.5
                potential += - self.masses[i] * self.masses[j] / r_ij
        return potential

    def kinetic(self, p_vec):
        p1 = p_vec[:, :2]
        p2 = p_vec[:, 2:4]
        p3 = p_vec[:, 4:]
        kinetic = .5 * (self.masses[0] * (p1**2).sum(1) + self.masses[1] * (p2**2).sum(1) + self.masses[2] * (p3**2).sum(1))
        return kinetic

    def forward(self, p_vec, q_vec):
        return self.potential(q_vec) + self.kinetic(p_vec)

    def calculate_time_derivative(self, t, z_vec_npy):
        dtype = torch.float32
        dim = 6
        z_vec = torch.as_tensor(z_vec_npy, dtype=dtype).view(-1, dim * 2)
        p_vec = z_vec[:, :dim]
        q_vec = z_vec[:, dim:]
        hamilt = self.forward(p_vec.requires_grad_(), q_vec.requires_grad_())
        time_derivative = torch.cat((-grad(hamilt.sum(), q_vec, create_graph=True)[0], grad(hamilt.sum(), p_vec, create_graph=True)[0]), 1)
        time_derivative_npy = time_derivative.detach().numpy().reshape(z_vec_npy.shape)
        return time_derivative_npy


## The next two functions are taken directly from https://github.com/greydanus/hamiltonian-nn/blob/master/experiment-3body/data.py, so that we have
## the same configurations for the three body experiments 
def rotate2d(p, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],[s, c]])
    R = np.transpose(R)
    return p.dot(R)


def random_config(n, nu=2e-1, min_radius=0.9, max_radius=1.2, return_tensors=True):
    masses = np.ones(3) * 1

    q1 = np.zeros([n, 2])

    q1 = 2*np.random.rand(n, 2) - 1
    r = np.random.rand(n) * (max_radius-min_radius) + min_radius

    ratio = r/np.sqrt(np.sum((q1**2), axis=1))
    q1 *= np.tile(np.expand_dims(ratio, 1), (1, 2))
    q2 = rotate2d(q1, theta=2*np.pi/3)
    q3 = rotate2d(q2, theta=2*np.pi/3)

    # # velocity that yields a circular orbit
    v1 = rotate2d(q1, theta=np.pi/2)
    v1 = v1 / np.tile(np.expand_dims(r**1.5, axis=1), (1, 2))
    v1 = v1 * np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2)) # scale factor to get circular trajectories
    v2 = rotate2d(v1, theta=2*np.pi/3)
    v3 = rotate2d(v2, theta=2*np.pi/3)

    # make the circular orbits slightly chaotic
    v1 *= 1 + nu*(2*np.random.rand(2) - 1)
    v2 *= 1 + nu*(2*np.random.rand(2) - 1)
    v3 *= 1 + nu*(2*np.random.rand(2) - 1)

    q = np.zeros([n, 6])
    p = np.zeros([n, 6])

    q[:, :2] = q1
    q[:, 2:4] = q2
    q[:, 4:] = q3
    p[:, :2] = v1 * masses[0]
    p[:, 2:4] = v2 * masses[1]
    p[:, 4:] = v3 * masses[2]

    if return_tensors:
        dtype = torch.float32
        masses = torch.as_tensor(masses, dtype=dtype)
        p = torch.as_tensor(p, dtype=dtype)
        q = torch.as_tensor(q, dtype=dtype)

    return masses, p, q


def simulate_3body_dynamics(n, T, dt):
    masses, p_0, q_0 = random_config(n, return_tensors=False)
    ThreeBody = ThreeBodyHamilt(masses)
    simulated_data = []
    for sample_index in range(n):    
        z_0 = np.concatenate((p_0[sample_index, :], q_0[sample_index, :]), axis=0)
        t_eval = np.arange(T) * dt
        sol = solve_ivp(fun=ThreeBody.calculate_time_derivative, y0=z_0, t_span=(0, T*dt), t_eval=t_eval, rtol=1e-4, atol=1e-7, method='RK45')
        t = sol.t
        trajectory = np.expand_dims(sol.y, 1)
        simulated_data.append(trajectory)
    simulated_data = np.concatenate(simulated_data, 1).transpose(2, 1, 0)
    simulated_data = np.float32(simulated_data)
    print ('dataset shape', simulated_data.shape)
    return simulated_data


def simulate_chain_dynamics(n=2000, T=20, dt=0.01, chain_length = 10, k=5, m=1, coarsening_factor=10):

    SprChain = SprChainHamilt(k, m)
    H = lambda p, q: SprChain(p, q)
    p_0 = (torch.rand(n, chain_length) - 0.5) * 5
    q_0 = (torch.rand(n, chain_length) - 0.5) * 5
    p_0[:, 0] = 0
    p_0[:, -1] = 0
    q_0[:, 0] = 0
    q_0[:, -1] = 0

    p_0.to(device)
    q_0.to(device)

    integrator = 'leapfrog'
    simulated_data = numerically_integrate(integrator=integrator, p_0=p_0, q_0=q_0, model=H, method=5, \
        T=T, dt=dt, volatile=True, device=device, coarsening_factor=coarsening_factor)
    simulated_data = simulated_data.cpu().numpy()

    print ('dataset shape', simulated_data.shape)

    return simulated_data
    

def generate_chain_datasets(dataset_index, T_train, T_test, n_samples_train, n_samples_test, chain_length, dt=0.1, coarsening_factor=10):
    data_dir = './data/3body'
    if (not os.path.isdir(data_dir)):
        os.mkdir(data_dir)
    k_true = torch.clamp(torch.randn(chain_length + 1) / 4 + 1, min=0.2) * 5
    m_true = torch.clamp(torch.randn(chain_length) / 4 + 1, min=0.2)
    train_data_chain = simulate_chain_dynamics(n=n_samples_train, T=T_train, dt=dt, k=k_true, m = m_true, chain_length = chain_length, coarsening_factor=coarsening_factor)
    np.save('./data/chain/train_data_chain_' + str(dataset_index), train_data_chain)
    test_data_chain = simulate_chain_dynamics(n=n_samples_test, T=T_test, dt=dt, k=k_true, m = m_true, chain_length = chain_length, coarsening_factor=coarsening_factor)
    np.save('./data/chain/test_data_chain_' + str(dataset_index), test_data_chain)


def generate_3body_datasets(dataset_index, T_train, T_test, n_samples_train, n_samples_test, dt=0.1, coarsening_factor=10):
    data_dir = './data/3body'
    if (not os.path.isdir(data_dir)):
        os.mkdir(data_dir)
    train_data_3body = simulate_3body_dynamics(n=n_samples_train, T=T_train, dt=dt)
    np.save(data_dir + '/train_data_3body_' + str(dataset_index), train_data_3body)
    test_data_3body = simulate_3body_dynamics(n=n_samples_test, T=T_test, dt=dt)
    np.save(data_dir + '/test_data_3body_' + str(dataset_index), test_data_3body)


def add_noise_chain(dataset_index, coarsening_factor, noise_level):
    train_data = np.load('./data/chain/train_data_chain_' + str(dataset_index) + '.npy')
    noise = torch.randn(train_data.shape).numpy() * noise_level
    train_data += noise
    np.save('./data/chain/train_data_chain_' + str(dataset_index) + '_n' + str(noise_level), train_data)
    test_data = np.load('./data/chain/test_data_chain_' + str(dataset_index) + '.npy')
    noise = torch.randn(test_data.shape).numpy() * noise_level
    test_data += noise
    np.save('./data/chain/test_data_chain_' + str(dataset_index) + '_n' + str(noise_level), test_data)


def main():
    dataset_index = 'newA'
    T = 100
    T_test = 50
    dt = 0.01
    n_samples = 100  ## number of training trajectories
    n_samples_test = 150
    coarsening_factor=1
    chain_length= 20

    ## generating the training and testing datasets for the chain experiments
    generate_3body_datasets(dataset_index=dataset_index, T_train=T, T_test=T_test, n_samples_train=n_samples, n_samples_test=n_samples_test, dt=dt)
    
    ## generating the training and testing datasets for the 3body experiments
    generate_chain_datasets(dataset_index=dataset_index, T_train=T, T_test=T_test, n_samples_train=n_samples, n_samples_test=n_samples_test, \
        chain_length=chain_length, dt=dt, coarsening_factor=coarsening_factor)
    add_noise_chain(dataset_index=dataset_index, coarsening_factor=coarsening_factor, noise_level=0.2)


    ######## resimulate 3body
    # test_data=np.load('./data/3body/test_data_3body_' + dataset_index + '.npy')
    # p_0 = torch.as_tensor(test_data[0, :, :6], dtype=torch.float32)
    # q_0 = torch.as_tensor(test_data[0, :, 6:], dtype=torch.float32)
    # masses = np.ones(3) * 1
    # config = masses, p_0, q_0
    # resimulated_test_data = create_3body_dataset(T=100, n_samples=test_data.shape[1], dt=1, coarsening_factor=1, config=config)
    # np.save('./data/3body/test_data_3body_' + dataset_index + '_resim1.npy', resimulated_test_data.numpy())


if __name__ == "__main__":
    main()
