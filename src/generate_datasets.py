import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn
import time
import random
from torch.autograd import Variable, grad
from sklearn.metrics import mean_squared_error
from scipy.ndimage.filters import gaussian_filter1d as gf1
from scipy.stats import norm
from scipy.integrate import solve_ivp

import train_and_predict, train_and_predict_image, train_and_predict_chain


if torch.cuda.is_available():
    device = torch.device('cuda:0')

## The real Hamiltonian used for simulating the training data, which is unknown to the learning algorithm
class SprLatHamilt(nn.Module):
    def __init__(self, n_v, n_h, k_v, k_h, m, fixed_bdr=False):
        super(SprLatHamilt, self).__init__()
        self.n_v = n_v
        self.n_h = n_h
        #self.m_inv = nn.Parameter(torch.ones(n_v, n_h) * m)
        self.m_inv = Variable(1 / m, requires_grad=False)
        #self.k_v = nn.Parameter(torch.ones(n_v-1, n_h) * k)
        #self.k_h = nn.Parameter(torch.ones(n_v, n_h-1) * k)
        self.k_v = Variable(k_v, requires_grad=False)
        self.k_h = Variable(k_h, requires_grad=False)
        self.fixed_bdr = fixed_bdr

    def forward(self, p_vec, q_vec):
        p, q = self.vec_to_mat(p_vec, q_vec, self.fixed_bdr)
        kinetic = 0.5 * (p.pow(2) * self.m_inv).sum(dim=1).sum(dim=1)
        q_diff_v = q[:, :-1, :] - q[:, 1:, :]
        q_diff_h = q[:, :, :-1] - q[:, :, 1:]
        potential = 0.5 * (self.k_v * q_diff_v.pow(2)).sum(dim=1).sum(dim=1) + 0.5 * (self.k_h * q_diff_h.pow(2)).sum(dim=1).sum(dim=1)
        return kinetic + potential

    def vec_to_mat(self, p_vec, q_vec, padding):
        if(padding):
            p_mat = Variable(torch.zeros(p_vec.shape[0], self.n_v, self.n_h), volatile=False)
            q_mat = Variable(torch.zeros(p_vec.shape[0], self.n_v + 2, self.n_h + 2), volatile=False)
            for row in range(self.n_v):
                p_mat[:, row, :] = p_vec[:, row * self.n_h:(row+1) * self.n_h]
                q_mat[:, row+1, 1:-1] = q_vec[:, row * self.n_h:(row+1) * self.n_h]
        else:
            p_mat = Variable(torch.Tensor(p_vec.shape[0], self.n_v, self.n_h), volatile=False)
            q_mat = Variable(torch.Tensor(p_vec.shape[0], self.n_v, self.n_h), volatile=False)
            for row in range(self.n_v):
                p_mat[:, row, :] = p_vec[:, row * self.n_h:(row+1) * self.n_h]
                q_mat[:, row, :] = q_vec[:, row * self.n_h:(row+1) * self.n_h]
        return p_mat, q_mat


class SprChainHamilt(nn.Module):
    def __init__(self, n, k, m):
        super(SprChainHamilt, self).__init__()
        self.n = n
        # self.m_inv = 1 / m
        self.m_inv = Variable(1 / m, requires_grad=False)
        # self.k = k
        self.k = Variable(k, requires_grad=False)

    def forward(self, p_vec, q_vec):
        p, q = p_vec, q_vec
        # kinetic = 0.5 * (p[:, 1:(p_vec.shape[1]-1)].pow(2) * self.m_inv).sum(dim=1) + 0.00005 * p[:, 0].pow(2) + 0.00005 * p[:, -1].pow(2)
        kinetic = 0.5 * (p.pow(2) * self.m_inv).sum(dim=1)
        q_diff = q[:, :-1] - q[:, 1:]
        potential = 0.5 * (self.k[1:-1] * q_diff.pow(2)).sum(dim=1) + 0.5 * (self.k[0] * q[:, 0].pow(2)) + 0.5 * (self.k[-1] * q[:, -1].pow(2))
                    # + (q[:, 0].pow(2) + q[:, -1].pow(2)) * 10000
        return kinetic + potential


class SprChainHamilt(nn.Module):
    def __init__(self, n, k, m):
        super(SprChainHamilt, self).__init__()
        self.n = n
        # self.m_inv = 1 / m
        self.m_inv = Variable(1 / m, requires_grad=False)
        # self.k = k
        self.k = Variable(k, requires_grad=False)

    def forward(self, p_vec, q_vec):
        p, q = p_vec, q_vec
        # kinetic = 0.5 * (p[:, 1:(p_vec.shape[1]-1)].pow(2) * self.m_inv).sum(dim=1) + 0.00005 * p[:, 0].pow(2) + 0.00005 * p[:, -1].pow(2)
        kinetic = 0.5 * (p.pow(2) * self.m_inv).sum(dim=1)
        q_diff = q[:, :-1] - q[:, 1:]
        potential = 0.5 * (self.k[1:-1] * q_diff.pow(2)).sum(dim=1) + 0.5 * (self.k[0] * q[:, 0].pow(2)) + 0.5 * (self.k[-1] * q[:, -1].pow(2))
                    # + (q[:, 0].pow(2) + q[:, -1].pow(2)) * 10000
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
        # print (potential.dtype)
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
        # time_derivative = torch.cat((-grad(hamilt.sum(), q_vec, create_graph=True)[0], p_vec), 0)
        time_derivative_npy = time_derivative.detach().numpy().reshape(z_vec_npy.shape)
        return time_derivative_npy


def rotate2d(p, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],[s, c]])
    # return (R.dot(p.reshape(2,1))).squeeze()
    # return (R.dot(np.transpose(p)))#.squeeze()
    R = np.transpose(R)
    return p.dot(R)

def random_config(n, nu=2e-1, min_radius=0.9, max_radius=1.2, return_tensors=True):
    '''This is not principled at all yet'''
    # state = np.zeros((3,5))
    # state[:,0] = 1
    masses = np.ones(3) * 1

    q1 = np.zeros([n, 2])

    q1 = 2*np.random.rand(n, 2) - 1
    r = np.random.rand(n) * (max_radius-min_radius) + min_radius

    # print (np.sqrt( np.sum((q1**2), axis=1) ).shape)
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

    # state[0,1:3], state[0,3:5] = p1, v1
    # state[1,1:3], state[1,3:5] = p2, v2
    # state[2,1:3], state[2,3:5] = p3, v3

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


def create_dataset_numpy(system_type, n, T, dt):
    if (system_type == 6):
        masses, p_0, q_0 = random_config(n, return_tensors=False)
        # print ('p_0', p_0)
        # print ('q_0', q_0)
        z_0 = np.concatenate((p_0, q_0), axis=1)
        ThreeBody = ThreeBodyHamilt(masses)
        # H = lambda p, q: ThreeBody(p, q)
        # n_steps = math.floor(T / dt) +1
        t_eval = np.arange(T) * dt
        # print ('teval', t_eval)
        # print ('z_0', z_0)
        sol = solve_ivp(fun=ThreeBody.calculate_time_derivative, y0=z_0.reshape(n * 12), t_span=(0, T*dt), t_eval=t_eval, rtol=1e-7, atol=1e-8, method='RK45')
        t = sol.t
        trajectories = np.transpose(sol.y).reshape(T, n, 12)
        # print ('t', t)
        print ('traj ivp',  trajectories.shape)

    return trajectories

def create_dataset_numpy_1by1(system_type, n, T, dt):
    if (system_type == 6):
        masses, p_0, q_0 = random_config(n, return_tensors=False)
        ThreeBody = ThreeBodyHamilt(masses)
        trajectories = []
        for sample_index in range(n):    
            z_0 = np.concatenate((p_0[sample_index, :], q_0[sample_index, :]), axis=0)
            # H = lambda p, q: ThreeBody(p, q)
            # n_steps = math.floor(T / dt) +1
            t_eval = np.arange(T) * dt
            # print ('teval', t_eval)
            # print ('z_0', z_0)
            sol = solve_ivp(fun=ThreeBody.calculate_time_derivative, y0=z_0, t_span=(0, T*dt), t_eval=t_eval, rtol=1e-4, atol=1e-7, method='RK45')
            t = sol.t
            trajectory = np.expand_dims(sol.y, 1)
            # print ('t', t)
            # print ('traj ivp',  trajectories.shape)
            print ('traj shape', trajectory.shape)
            trajectories.append(trajectory)
        trajectories = np.concatenate(trajectories, 1).transpose(2, 1, 0)
        print ('trajs shape', trajectories.shape)
    return trajectories



# class Particle2Dpotential(nn.Module):
#     def __init__(self, m, k):


## simulates the training data - the trajectories of p's and q's for a spring lattice
def create_dataset(system_type, leap=True, n=2000, T=20, dt=0.01, n_v = 5, n_h = 5, \
 chain_length = 10, k=5, k_v=5, k_h=5, m=1, coarsen = True, fixed_bdr = False, r=1.0, \
 p_max=0.5, p_min=0.2, q_max=0.4, q_min=0.2, n_particles=1, coarsening_factor=10):
    # coarsening_factor = 10
    if (system_type == 0):
        H = lambda p, q: 0.5 * p.pow(2).sum(dim=1, keepdim=True) + k * 0.5 * q.pow(2).sum(dim=1, keepdim=True)
        p_0 = (torch.rand(n, 1) - 0.5) * 2
        q_0 = (torch.rand(n, 1) - 0.5) * 2
    elif (system_type == 1):
        g = 9.8
        H = lambda p, q: 0.5 * p.pow(2) / m - g * q.cos()
        p_0 = (torch.rand(n, 1) - 0.5) * 2
        q_0 = (torch.rand(n, 1) - 0.5) * 2
    elif (system_type == 2):
        SprLat = SprLatHamilt(n_v, n_h, k_v, k_h, m, fixed_bdr)
        H = lambda p, q: SprLat(p, q)
        p_0 = (torch.rand(n, n_v * n_h) - 0.5) * 2
        q_0 = (torch.rand(n, n_v * n_h) - 0.5) * 2
    elif (system_type == 3):
        SprChain = SprChainHamilt(chain_length, k, m)
        H = lambda p, q: SprChain(p, q)
        p_0 = (torch.rand(n, chain_length) - 0.5) * 5
        q_0 = (torch.rand(n, chain_length) - 0.5) * 5
        p_0[:, 0] = 0
        p_0[:, -1] = 0
        q_0[:, 0] = 0
        q_0[:, -1] = 0
    elif (system_type == 4):
        H = lambda p, q: 0.5 * p.pow(2).sum(dim=1, keepdim=True) - r / q.pow(2).sum(dim=1, keepdim=True).sqrt()
        # p_0 = (torch.rand(n, 2 * n_particles) - 0.5) * 2 * p_scale
        # q_0 = (torch.rand(n, 2 * n_particles) - 0.5) * 2 * q_scale
        theta_0 = (np.random.rand(n, n_particles) - 0.5) * 2 * math.pi
        # p_0_norm = np.random.randn(n, n_particles) * p_scale
        p_0_norm = np.random.rand(n, n_particles) * (p_max - p_min) + p_min
        # q_0_norm = np.random.randn(n, n_particles) * q_scale
        q_0_norm = np.random.rand(n, n_particles) * (q_max - q_min) + q_min
        p_0_npy = np.concatenate([p_0_norm * np.cos(theta_0), p_0_norm * np.sin(theta_0)], 1)
        clockwise = np.random.binomial(1, 0.5, [n, n_particles]) * 2 - 1
        q_0_npy = np.concatenate([q_0_norm * - np.sin(theta_0) * clockwise, q_0_norm * np.cos(theta_0) * clockwise], 1)
        p_0 = torch.from_numpy(p_0_npy)
        q_0 = torch.from_numpy(q_0_npy)

    elif (system_type == 5):
        H = lambda p, q: 0.5 * p.pow(2).sum(dim=1, keepdim=True) + r * q.pow(2).sum(dim=1, keepdim=True)
        p_0 = (torch.rand(n, 2 * n_particles) - 0.5) * 2
        q_0 = (torch.rand(n, 2 * n_particles) - 0.5) * 2

    elif (system_type == 6):
        masses, p_0, q_0 = random_config(n)
        ThreeBody = ThreeBodyHamilt(masses)
        H = lambda p, q: ThreeBody(p, q)

    # cuda = torch.cuda.is_available()

    # if cuda:
    #     # H = H.cuda()
    #     p_0 = p_0.cuda()
    #     q_0 = q_0.cuda()
    p_0.to(device)
    q_0.to(device)

    if leap:
        if coarsen:
            # if cuda:
            #     fine_data = train_and_predict.leapfrog(p_0, q_0, H, T * coarsening_factor, dt / coarsening_factor, volatile=True).to(torch.device('cpu')).data
            # else:
            #     fine_data = train_and_predict.leapfrog(p_0, q_0, H, T * coarsening_factor, dt / coarsening_factor, volatile=True).data
            fine_data = train_and_predict.leapfrogG(p_0, q_0, H, T * coarsening_factor, dt / coarsening_factor, volatile=True, use_tqdm=True).to(torch.device('cpu')).data
            coarse_data = fine_data[np.arange(T) * coarsening_factor, :, :]
            return coarse_data
        else:
            # if cuda:
            #     return train_and_predict.leapfrog(p_0, q_0, H, T, dt, volatile=True).to(torch.device('cpu')).data
            # else:
            #     return train_and_predict.leapfrog(p_0, q_0, H, T, dt, volatile=True).data
            return train_and_predict.leapfrogG(p_0, q_0, H, T, dt, volatile=True, use_tqdm=True).to(torch.device('cpu')).data
    else:
        ## This is the case where we can solve for the true trajectory of the system analytically. For most of the systems this is not applicable.
        p_0 = (torch.rand(n, 1) - 0.5) * 2
        q_0 = (torch.rand(n, 1) - 0.5) * 2
        trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]),
            volatile=True)

        sqrt_k = np.sqrt(k)
        c_1 = q_0[0, 0] / sqrt_k
        c_2 = p_0[0, 0]

        def exact_p(t, sqrt_k, c_1, c_2):
            return c_1 * np.sin(sqrt_k * t) + c_2 * np.cos(sqrt_k * t)

        def exact_q(t, sqrt_k, c_1, c_2):
            return (c_1 * sqrt_k * np.cos(sqrt_k * t) - c_2 * sqrt_k * np.sin(sqrt_k * t)) / (-k)

        t_vec = np.arange(T) * dt
        p_vec = exact_p(t_vec, k, c_1, c_2)
        q_vec = exact_q(t_vec, k, c_1, c_2)

        for i in range(T):
             trajectories[i, :, :p_0.shape[1]] = p_vec[i]
             trajectories[i, :, p_0.shape[1]:] = q_vec[i]
        return trajectories.data

def create_3body_dataset(T, n_samples, dt=0.1, coarsening_factor=10, config=None):
    if (config is None):
        masses, p_0, q_0 = random_config(n)
    else:
        masses, p_0, q_0 = config
    ThreeBody = ThreeBodyHamilt(masses)
    H = lambda p, q: ThreeBody(p, q)

    if (coarsening_factor > 1):
        fine_data = train_and_predict.leapfrogG(p_0, q_0, H, T * coarsening_factor, dt / coarsening_factor, volatile=True, use_tqdm=True).to(torch.device('cpu')).data
        coarse_data = fine_data[np.arange(T) * coarsening_factor, :, :]
        return coarse_data
    else:
        return train_and_predict.leapfrogG(p_0, q_0, H, T, dt, volatile=True, use_tqdm=True).to(torch.device('cpu')).data


def create_combined_dataset_lattice(dataset_index, T_total, n_samples, n_v, n_h, dt=0.1):
    system_type = 2
    k_v_true = torch.clamp(torch.randn([n_v+1, n_h+2]) / 4 + 1, min=0.2) * 5
    k_h_true = torch.clamp(torch.randn([n_v+2, n_h+1]) / 4 + 1, min=0.2) * 5
    m_true = torch.clamp(torch.randn([n_v, n_h]) / 4 + 1, min=0.2)
    combined_data = create_dataset(system_type, leap=True, n=n_samples, T=T_total, dt=dt, k_v = k_v_true, k_h=k_h_true, m = m_true, n_v = n_v, n_h = n_h, fixed_bdr = True)
    np.save('./data/combined_data_3by5fb_' + str(dataset_index), combined_data.numpy())
    np.save('./data/m_3by5fb_' + str(dataset_index), m_true.numpy())
    np.save('./data/k_v_3by5fb_' + str(dataset_index), k_v_true.numpy())
    np.save('./data/k_h_3by5fb_' + str(dataset_index), k_h_true.numpy())

def create_combined_dataset_chain(dataset_index, T_total, n_samples, chain_length, dt=0.1):
    system_type = 3
    k_true = torch.clamp(torch.randn(chain_length - 1) / 4 + 1, min=0.2) * 5
    # k_true = torch.clamp(torch.randn(chain_length + 1) / 4 + 1, min=0.2) * 5
    m_true = torch.clamp(torch.randn(chain_length) / 4 + 1, min=0.2)
    combined_data_chain = create_dataset(system_type, leap=True, n=n_samples, T=T_total, dt=dt, k=k_true, m = m_true, chain_length = chain_length)
    np.save('./data/combined_data_sc20_' + str(dataset_index), combined_data_chain.numpy())
    np.save('./data/m_sc20_' + str(dataset_index), m_true.numpy())
    np.save('./data/k_sc20_' + str(dataset_index), k_true.numpy())

def create_train_test_datasets_chain(dataset_index, T_train, T_test, n_samples_train, n_samples_test, chain_length, dt=0.1, coarsening_factor=10):
    system_type = 3
    # k_true = torch.clamp(torch.randn(chain_length - 1) / 4 + 1, min=0.2) * 5
    k_true = torch.clamp(torch.randn(chain_length + 1) / 4 + 1, min=0.2) * 5
    m_true = torch.clamp(torch.randn(chain_length) / 4 + 1, min=0.2)
    train_data_chain = create_dataset(system_type, leap=True, n=n_samples_train, T=T_train, dt=dt, k=k_true, m = m_true, chain_length = chain_length, coarsening_factor=coarsening_factor)
    np.save('./data/chain/train_data_chain_' + str(dataset_index) + '_cf' + str(coarsening_factor), train_data_chain.numpy())
    test_data_chain = create_dataset(system_type, leap=True, n=n_samples_test, T=T_test, dt=dt, k=k_true, m = m_true, chain_length = chain_length, coarsening_factor=coarsening_factor)
    np.save('./data/chain/test_data_chain_' + str(dataset_index) + '_cf' + str(coarsening_factor), test_data_chain.numpy())
    # np.save('./data/m_sc20_' + str(dataset_index), m_true.numpy())
    # np.save('./data/k_sc20_' + str(dataset_index), k_true.numpy())
    print ('yes + 1')

def create_train_test_datasets_three(dataset_index, T_train, T_test, n_samples_train, n_samples_test, dt=0.1, coarsening_factor=10):
    system_type = 6
    # k_true = torch.clamp(torch.randn(chain_length - 1) / 4 + 1, min=0.2) * 5
    # train_data_3body = create_dataset(system_type, leap=True, n=n_samples_train, T=T_train, dt=dt, coarsening_factor=coarsening_factor)
    # train_data_3body = create_dataset_numpy(system_type, n=n_samples_train, T=T_train, dt=dt)
    train_data_3body, masses = create_3body_dataset(T_train, n_samples_train, dt=dt, coarsening_factor=coarsening_factor, config=None)
    np.save('./data/chain/train_data_3body_' + str(dataset_index) + '_cf' + str(coarsening_factor), train_data_3body.numpy())

    test_data_3body = create_dataset(system_type, leap=True, n=n_samples_test, T=T_test, dt=dt, coarsening_factor=coarsening_factor)
    # test_data_3body = create_dataset_numpy(system_type, n=n_samples_test, T=T_test, dt=dt)
    np.save('./data/chain/test_data_3body_' + str(dataset_index) + '_cf' + str(coarsening_factor), test_data_3body.numpy())
    # np.save('./data/m_sc20_' + str(dataset_index), m_true.numpy())
    # np.save('./data/k_sc20_' + str(dataset_index), k_true.numpy())
    print ('yes + 1')

def create_train_test_datasets_three_numpy(dataset_index, T_train, T_test, n_samples_train, n_samples_test, dt=0.1, coarsening_factor=10):
    data_dir = './data/3body'
    if (not os.path.isdir(data_dir)):
        os.mkdir(data_dir)
    system_type = 6
    # k_true = torch.clamp(torch.randn(chain_length - 1) / 4 + 1, min=0.2) * 5
    # train_data_chain = create_dataset(system_type, leap=True, n=n_samples_train, T=T_train, dt=dt, coarsening_factor=coarsening_factor)
    train_data_3body = create_dataset_numpy_1by1(system_type, n=n_samples_train, T=T_train, dt=dt)
    np.save(data_dir + '/train_data_3body_' + str(dataset_index) + '_ivp', train_data_3body)
    # test_data_chain = create_dataset(system_type, leap=True, n=n_samples_test, T=T_test, dt=dt, coarsening_factor=coarsening_factor)
    test_data_3body = create_dataset_numpy_1by1(system_type, n=n_samples_test, T=T_test, dt=dt)
    np.save(data_dir + '/test_data_3body_' + str(dataset_index) + '_ivp', test_data_3body)
    # np.save('./data/m_sc20_' + str(dataset_index), m_true.numpy())
    # np.save('./data/k_sc20_' + str(dataset_index), k_true.numpy())


def create_combined_dataset_pendulum(dataset_index, T_total, n_samples, dt=0.1):
    system_type = 1
    # m_true = torch.rand(1)
    m_true = random.random()
    combined_data_pendulum = create_dataset(system_type, leap=True, n=n_samples, T=T_total, dt=dt, m = m_true)
    np.save('./data/combined_data_pend_' + str(dataset_index), combined_data_pendulum.numpy())
    np.save('./data/m_pend_' + str(dataset_index), m_true)

def ends_to_image(left_end, right_end, n_window, window_size):
    image = np.zeros(n_window)
    left_adjusted = left_end + n_window * window_size / 2
    right_adjusted = right_end + n_window * window_size / 2
    leftmost_window = (np.ceil((left_end + n_window * window_size / 2) / window_size)).astype(int)
    rightmost_window = (np.floor((right_end + n_window * window_size / 2) / window_size)).astype(int)
    image[max(leftmost_window, 0) : min(rightmost_window, n_window)] = 1
    if (leftmost_window > 0):
        image[leftmost_window - 1] = window_size - left_adjusted % window_size
    if (rightmost_window < n_window):
        image[rightmost_window] = right_adjusted % window_size
    return image

def slit_image_dataset_from_pend_dataset(dataset_index, d, h, l, n_window, window_size, smooth=False, sigma=1):
    combined_data_pendulum = np.load('./data/combined_data_pend_' + str(dataset_index) + '.npy')
    q_traj = combined_data_pendulum[:, :, 1]
    mid_traj = h * np.tan(q_traj)
    left_end_traj = mid_traj - d / np.cos(q_traj)
    right_end_traj = mid_traj + d / np.cos(q_traj)
    slit_image_dataset = np.zeros([combined_data_pendulum.shape[0], combined_data_pendulum.shape[1], n_window])
    for i in range(combined_data_pendulum.shape[0]):
        for j in range(combined_data_pendulum.shape[1]):
            slit_image_dataset[i, j, :] = ends_to_image(left_end_traj[i, j], right_end_traj[i, j], n_window, window_size)
    if smooth:
        slit_image_dataset_gs = gf1(slit_image_dataset, sigma=sigma, axis=2)
        np.save('./data/slit_image_data_pend_' + str(dataset_index) + '_nw' + str(n_window) + '_gs' + str(sigma), slit_image_dataset_gs)
        return slit_image_dataset_gs
    else:
        np.save('./data/slit_image_data_pend_' + str(dataset_index) + '_nw' + str(n_window), slit_image_dataset)
        return slit_image_dataset

def generate_motion_data(dataset_index, smooth=False, n_window=30, sigma=0.3):
    if smooth:
        slit_image_data = np.load('./data/slit_image_data_pend_' + str(dataset_index) + '_nw' + str(n_window) + '_gs' + str(sigma) + '.npy')
    else:
        slit_image_data = np.load('./data/slit_image_data_pend_' + str(dataset_index) + '_nw' + str(n_window) + '.npy')
    slit_image_diff_data = slit_image_data[2:, :, :] - slit_image_data[:-2]
    if smooth:
        np.save('./data/slit_image_diff_data_pend_' + str(dataset_index) + '_nw' + str(n_window) + '_gs' + str(sigma), slit_image_diff_data)
    else:
        np.save('./data/slit_image_diff_data_pend_' + str(dataset_index) + '_nw' + str(n_window), slit_image_diff_data)

def create_combined_dataset_1p2d(dataset_index, T_total, n_samples, r=0.05, p_max=0.5, p_min=0.2, q_max=0.4, q_min=0.2, dt=0.1, coarsening_factor=10):
    system_type = 4
    # m_true = torch.rand(1)
    # r_true = 0.8
    # r = 0.05
    # p_scale = 0.1
    # q_scale = 1
    combined_data_1p2d = create_dataset(system_type, leap=True, n=n_samples, T=T_total, dt=dt, r=r, p_max=p_max, p_min=p_min, q_max=q_max, q_min=q_min, coarsening_factor=coarsening_factor)
    np.save('./data/combined_data_1p2d_' + str(dataset_index) + '_dt' + str(dt) +  '_cf' + str(coarsening_factor) + '_newlf', combined_data_1p2d.numpy())
    # np.save('./data/m_1p2d_' + str(dataset_index), m_true)

def create_combined_dataset_1p2dqp(dataset_index, T_total, n_samples, dt=0.1, coarsening_factor=10):
    system_type = 5
    # m_true = torch.rand(1)
    r_true = 1.0
    combined_data_1p2dqp = create_dataset(system_type, leap=True, n=n_samples, T=T_total, dt=dt, r=r_true, coarsening_factor=coarsening_factor)
    np.save('./data/combined_data_1p2dqp_' + str(dataset_index) + '_cf' + str(coarsening_factor) + '_newlf', combined_data_1p2dqp.numpy())

def animate_1p2d(traj, traj_true=None, interval=200):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    particle, = ax.plot([], [], '^', lw=2)
    # if (traj_true != None):
    #     particle_2, = ax.plot([], [], '^', lw=2)

    # for k in range(T):
    #     wframe = ax.plot(traj[k, 0], traj[k, 1])
    #     plt.pause(0.1)
    #     ax.remove(wframe)

    def init():
        particle.set_data([], [])
        return particle,

    def animate(i):
        x = traj[i, 2]
        y = traj[i, 3]
        if (traj_true is None):
            particle.set_data(np.array([x]), np.array([y]))
        else:
            x_2 = traj_true[i, 2]
            y_2 = traj_true[i, 3]
            particle.set_data(np.array([x, x_2]), np.array([y, y_2]))
        # if (traj_true != None):
            # x = traj_true[i, 2]
            # y = traj_true[i, 3]
            # particle2.set_data(np.array([x]), np.array([y]))
        return particle,
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=traj.shape[0], interval=interval, blit=True)
    plt.show()

def position_to_image(q, pixel_size_v, pixel_size_h, pixel_num_v, pixel_num_h, sigma, cutoff):
    image = np.zeros([pixel_num_v, pixel_num_h])
    center_v = (q[0] + (pixel_num_v * pixel_size_v / 2)) / pixel_size_v
    center_h = (q[1] + (pixel_num_h * pixel_size_h / 2)) / pixel_size_h
    center_v_index = int(center_v)
    center_h_index = int(center_h)
    for i in range(int(np.ceil(cutoff / pixel_size_v))):
        # bottom_v = center_v + pixel_num_v / 2 - i
        # top_v = center_v + pixel_num_v / 2 + i
        bottom_v = center_v - i
        top_v = center_v + i
        bottom_v_index = int(bottom_v)
        top_v_index = int(top_v)
        # print ('i', i)
        cutoff_i = np.sqrt(pow(cutoff, 2) - pow(i * pixel_size_v, 2))
        if (bottom_v >= 0) & (bottom_v < pixel_num_v):
            for j in range(int(np.ceil(cutoff_i / pixel_size_h))):

                left_h = center_h - j
                right_h = center_h + j
                left_h_index = int(left_h)
                right_h_index = int(right_h)

                if (left_h >= 0) & (left_h < pixel_num_h):
                    image[bottom_v_index, left_h_index] = norm.pdf(np.sqrt(pow((center_v - bottom_v_index) * pixel_size_v, 2) + pow((center_h - left_h_index) * pixel_size_h, 2)) / sigma)
                if (right_h >= 0) & (right_h < pixel_num_h):
                    image[bottom_v_index, right_h_index] = norm.pdf(np.sqrt(pow((center_v - bottom_v_index) * pixel_size_v, 2) + pow((center_h - right_h_index) * pixel_size_h, 2)) / sigma)
        if (top_v >= 0) & (top_v < pixel_num_v):
            for j in range(int(np.ceil(cutoff_i / pixel_size_h))):
                left_h = center_h - j
                right_h = center_h + j
                left_h_index = int(left_h)
                right_h_index = int(right_h)
                if (left_h >= 0) & (left_h < pixel_num_h):
                    image[top_v_index, left_h_index] = norm.pdf(np.sqrt(pow((center_v - top_v_index) * pixel_size_v, 2) + pow((center_h - left_h_index) * pixel_size_h, 2)) / sigma)
                if (right_h >= 0) & (right_h < pixel_num_h):
                    image[top_v_index, right_h_index] = norm.pdf(np.sqrt(pow((center_v - top_v_index) * pixel_size_v, 2) + pow((center_h - right_h_index) * pixel_size_h, 2)) / sigma)
    return image


def get_image_dataset_1p2d(dataset_index, pixel_size_v, pixel_size_h, pixel_num_v, pixel_num_h, sigma, cutoff):
    # combined_data_1p2d = np.load('./data/combined_data_1p2d_' + str(dataset_index) + '.npy')
    # combined_data_1p2d = np.load('./data/combined_data_1p2dqp_' + str(dataset_index) + '.npy')
    # combined_data_1p2d = np.load('./data/combined_data_filtered_1p2d_' + str(dataset_index) + '.npy')
    # combined_data_1p2d = np.load('/home/chenzh/Documents/LBPY/learningphysics/src/data/combined_data_1p2d_A_cf1000_newlf.npy')
    T = combined_data_1p2d.shape[0]
    n_samples = combined_data_1p2d.shape[1]
    image_dataset = np.zeros([T, n_samples, pixel_num_v * pixel_num_h])
    # np.save('/data/chenzh/learningphysics/data/1p2d_image_data_' + str(dataset_index) + '_' + str(pixel_size_v) + '_' + str(pixel_num_v) + '_s' + str(sigma), image_dataset)
    # np.save('/data/chenzh/learningphysics/data/1p2dqp_image_data_' + str(dataset_index) + '_' + str(pixel_size_v) + '_' + str(pixel_num_v) + '_s' + str(sigma), image_dataset)
    np.save('/data/chenzh/learningphysics/data/1p2d_image_data_filtered' + str(dataset_index) + '_' + str(pixel_size_v) + '_' + str(pixel_num_v) + '_s' + str(sigma), image_dataset)


    for t in range(T):
        print ('t', t)
        for sample_index in range(n_samples):
            position = combined_data_1p2d[t, sample_index, 2:]
            image = position_to_image(position, pixel_size_v, pixel_size_h, pixel_num_v, pixel_num_h, sigma, cutoff)
            image_dataset[t, sample_index, :] = image.reshape(pixel_num_v * pixel_num_h)
    # np.save('/data/chenzh/learningphysics/data/1p2d_image_data_' + str(dataset_index) + '_' + str(pixel_size_v) + '_' + str(pixel_num_v) + '_s' + str(sigma), image_dataset)
    # np.save('/data/chenzh/learningphysics/data/1p2dqp_image_data_' + str(dataset_index) + '_' + str(pixel_size_v) + '_' + str(pixel_num_v) + '_s' + str(sigma), image_dataset)
    np.save('/data/chenzh/learningphysics/data/1p2d_image_data_filtered' + str(dataset_index) + '_' + str(pixel_size_v) + '_' + str(pixel_num_v) + '_s' + str(sigma), image_dataset)

    return image_dataset

def get_image_dataset_1p2d_mini(dataset_index, pixel_size_v, pixel_size_h, pixel_num_v, pixel_num_h, sigma, cutoff, T, n_samples):
    combined_data_1p2d = np.load('./data/combined_data_1p2d_' + str(dataset_index) + '.npy')
    # T = combined_data_1p2d.shape[0]
    # n_samples = combined_data_1p2d.shape[1]
    image_dataset = np.zeros([T, n_samples, pixel_num_v * pixel_num_h])
    for t in range(T):
        print ('t', t)
        for sample_index in range(n_samples):
            position = combined_data_1p2d[t, sample_index, 2:]
            image = position_to_image(position, pixel_size_v, pixel_size_h, pixel_num_v, pixel_num_h, sigma, cutoff)
            image_dataset[t, sample_index, :] = image.reshape(pixel_num_v * pixel_num_h)
    np.save('/data/chenzh/learningphysics/data/1p2d_image_data_mini_' + str(dataset_index) + '_' + str(pixel_size_v) + '_' + str(pixel_num_v), image_dataset)
    return image_dataset

def generate_motion_data_1p2d(dataset_index, pixel_size_v=0.1, pixel_size_h=0.1, pixel_num_v=32, pixel_num_h=32):
    # image_data = np.load('/data/chenzh/learningphysics/data/combined_data_1p2d_miniA_0.1_32.npy')
    image_data = np.load('/data/chenzh/learningphysics/data/1p2d_image_data_A_0.1_32_s0.15.npy')
    image_diff_data = image_data[2:, :, :] - image_data[:-2]
    # np.save('/data/chenzh/learningphysics/data/1p2d_image_diff_data_mini_' + str(dataset_index) + '_' + str(pixel_size_v) + '_' + str(pixel_num_v), image_diff_data)
    np.save('/data/chenzh/learningphysics/data/1p2d_image_diff_data_A_0.1_32_s0.15.npy', image_diff_data)
    return image_data

def filter_combined_data(dataset_index, system_type=4, r=0.8, thresh=100, dt=0.1):
    # combined_data = np.load('./data/combined_data_1p2d_' + str(dataset_index) + '_081718.npy')
    # combined_data = np.load('/home/chenzh/Documents/LBPY/learningphysics/src/data/combined_data_1p2d_' + str(dataset_index) + '_cf1000_newlf.npy')
    combined_data = np.load('/home/chenzh/Documents/LBPY/learningphysics/src/data/combined_data_1p2d_' + str(dataset_index) + '_dt' + str(dt) + '_cf1000_newlf.npy')
    if (system_type == 0):
        H = lambda p, q: 0.5 * p.pow(2).sum(dim=1, keepdim=True) + k * 0.5 * q.pow(2).sum(dim=1, keepdim=True)
    elif (system_type == 1):
        g = 9.8
        H = lambda p, q: 0.5 * p.pow(2) / m - g * q.cos()
    elif (system_type == 2):
        SprLat = SprLatHamilt(n_v, n_h, k_v, k_h, m, fixed_bdr)
        H = lambda p, q: SprLat(p, q)
    elif (system_type == 3):
        SprChain = SprChainHamilt(chain_length, k, m)
        H = lambda p, q: SprChain(p, q)
    elif (system_type == 4):
        H = lambda p, q: 0.5 * p.pow(2).sum(dim=1, keepdim=True) - r / q.pow(2).sum(dim=1, keepdim=True).sqrt()
    elif (system_type == 5):
        H = lambda p, q: 0.5 * p.pow(2).sum(dim=1, keepdim=True) + r * q.pow(2).sum(dim=1, keepdim=True)

    T_total = combined_data.shape[0]
    n_samples = combined_data.shape[1]
    dim = combined_data.shape[2] // 2
    hamilt_values = H(torch.from_numpy(combined_data[:, :, :dim].reshape(T_total * n_samples, dim)), torch.from_numpy(combined_data[:, :, dim:].reshape(T_total * n_samples, dim))).numpy().reshape(T_total, n_samples)
    hamilt_values_range = np.ptp(hamilt_values, axis=0)
    good_indices = np.where(hamilt_values_range <= thresh)[0]
    bad_indices = np.where(hamilt_values_range > thresh)[0]
    print ('good indices', good_indices)
    print ('num of good indices', len(good_indices))
    print ('bad indices', bad_indices)
    print ('num of bad indices', len(bad_indices))
    combined_data_filtered = combined_data[:, good_indices, :]
    np.save('./data/combined_data_filtered_1p2d_' + str(dataset_index) + '_th' + str(thresh) + '.npy', combined_data_filtered)
    # image_data_npy = np.load('/data/chenzh/learningphysics/data/1p2d_image_data_A_0.1_32_s0.15.npy')
    # image_diff_data_npy = np.load('/data/chenzh/learningphysics/data/1p2d_image_diff_data_A_0.1_32_s0.15.npy')
    # image_data_npy_filtered = image_data_npy[:, good_indices, :]
    # image_diff_data_npy_filtered = image_diff_data_npy[:, good_indices, :]
    # np.save('/data/chenzh/learningphysics/data/1p2d_image_data_filtered_A_0.1_32_s0.15_th' + str(thresh) + '.npy', image_data_npy_filtered)
    # np.save('/data/chenzh/learningphysics/data/1p2d_image_diff_data_filtered_A_0.1_32_s0.15_th' + str(thresh) + '.npy', image_diff_data_npy_filtered)
    return good_indices, bad_indices
def dist(combined_data):
    dist = np.power(combined_data[:, :, 2:], 2).sum(axis =2)
    return dist

def add_noise_chain(dataset_index, noise_level):
    train_data = np.load('./data/chain/train_data_chain_' + str(dataset_index) + '_cf100' + '.npy')
    noise = torch.randn(train_data.shape).numpy() * noise_level
    train_data += noise
    np.save('./data/chain/train_data_chain_' + str(dataset_index) + 'n' + str(noise_level), train_data)
    test_data = np.load('./data/chain/test_data_chain_' + str(dataset_index) + '_cf100' + '.npy')
    noise = torch.randn(test_data.shape).numpy() * noise_level
    test_data += noise
    np.save('./data/chain/test_data_chain_' + str(dataset_index) + 'n' + str(noise_level), test_data)


def main():
    dataset_index = 'F_01'
    T = 500  ## time length of training trajectories
    # T = 5000
    T_test = 500
    # T_test = 200  ## time length of testing trajectories
    # T_test = 15000
    # dt = 0.01
    dt = 0.1
    # dt = 0.001
    n_samples = 1000  ## number of training trajectories
    # n_samples = 100
    # n_samples = 1024
    # n_samples = 1000
    n_samples_test = 64
    # n_samples_test = 32
    # coarsening_factor = 30
    # coarsening_factor = 500
    # coarsening_factor = 10
    coarsening_factor=1

    ## Size of the spring lattice
    # n_v = 3  ## vertical size of the spring lattice
    # n_h = 5  ## horizontal size of the spring lattice

    ## Size of the spring chain
    chain_length= 20

    # r=0.05
    # p_max = 0.4
    # p_min = 0.2
    # q_max = 0.4
    # q_min = 0.2

    # create_combined_dataset_pendulum(dataset_index, T+T_test, n_samples, dt)
    # create_combined_dataset_chain(dataset_index, T+T_test, n_samples, chain_length, dt)
    # create_train_test_datasets_chain(dataset_index=dataset_index, T_train=T, T_test=T_test, n_samples_train=n_samples, n_samples_test=n_samples_test, chain_length=chain_length, dt=dt, coarsening_factor=coarsening_factor)
    # create_train_test_datasets_three(dataset_index=dataset_index, T_train=T, T_test=T_test, n_samples_train=n_samples, n_samples_test=n_samples_test, dt=dt, coarsening_factor=coarsening_factor)
    create_train_test_datasets_three_numpy(dataset_index=dataset_index, T_train=T, T_test=T_test, n_samples_train=n_samples, n_samples_test=n_samples_test, dt=dt)
    # create_combined_dataset_lattice(dataset_index, T+T_test, n_samples, n_v, n_h, dt)
    # create_combined_dataset_1p2d(dataset_index, T+T_test, n_samples, r=r, p_max=p_max, p_min=p_min, q_max=q_max, q_min=q_min, dt=dt, coarsening_factor=coarsening_factor)
    # create_combined_dataset_1p2dqp(dataset_index, T+T_test, n_samples, dt, coarsening_factor)


    # dataset_index = 'C'
    # d = 0.5
    # # h = 5
    # h = 3
    # # l = 20
    # l = 10
    # n_window = 30
    # window_size = 0.4

    # Gaussian_smoothing = False
    # sigma = 0.0

    # sid = slit_image_dataset_from_pend_dataset(dataset_index, d, h, l, n_window, window_size, smooth=Gaussian_smoothing, sigma=sigma)
    # generate_motion_data(dataset_index, False, 30, 0.0)

    ## 1p2d
    # dataset_index = 'J'
    # # pixel_size_v = 0.5
    # # pixel_size_v = 0.8
    # pixel_size_v = 0.1
    # # pixel_size_h = 0.5
    # # pixel_size_h = 0.8
    # pixel_size_h = 0.1
    # pixel_num_v = 32
    # pixel_num_h = 32
    # # sigma = 0.1
    # sigma = 0.15
    # # cutoff = 5
    # cutoff = 0.6
    # # get_image_dataset_1p2d(dataset_index, pixel_size_v, pixel_size_h, pixel_num_v, pixel_num_h, sigma, cutoff)
    # # get_image_dataset_1p2d_mini(dataset_index, pixel_size_v, pixel_size_h, pixel_num_v, pixel_num_h, sigma, cutoff, 600, 256)
    # # generate_motion_data_1p2d(dataset_index, pixel_size_v=0.1, pixel_size_h=0.1, pixel_num_v=32, pixel_num_h=32)
    # # filter_combined_data(dataset_index, system_type=4, r=0.8, thresh=1)
    # filter_combined_data(dataset_index, system_type=4, r=0.05, thresh=0.001, dt=dt)

    # add_noise_chain(dataset_index=dataset_index, noise_level=0.5)


    ######## resimulate 3body
    # test_data=np.load('./data/3body/test_data_3body_D_001_ivp.npy')
    # p_0 = torch.as_tensor(test_data[0, :, :6], dtype=torch.float32)
    # q_0 = torch.as_tensor(test_data[0, :, 6:], dtype=torch.float32)
    # masses = np.ones(3) * 1
    # config = masses, p_0, q_0
    # resimulated_test_data = create_3body_dataset(T=100, n_samples=test_data.shape[1], dt=1, coarsening_factor=1, config=config)
    # np.save('./data/3body/test_data_3body_D_001_ivp_resim1.npy', resimulated_test_data.numpy())



def show_animation_of_pendulum_slit_images(n_window, sigma):
    dataset_index = 'C'
    sample_index = 0
    sid = np.load('./data/slit_image_data_pend_' + str(dataset_index) + '_nw' + str(n_window) + '_gs' + str(sigma) + '.npy')
    train_and_predict_image.plot_chain_animation(n_window, sid[:, sample_index, :])

if __name__ == "__main__":
    main()
    # show_animation_of_pendulum_slit_images(30, 0.5)
