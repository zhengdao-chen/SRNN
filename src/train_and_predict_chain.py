import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn
import time
import random
from torch.autograd import Variable, grad
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pickle
import sys
sys.path.append('../src')
# import utils.Logger
from utils import Logger, loss_plot, loss_plot_restricted
from os import path
from train_and_predict import leapfrogG, eulerG


## calculates the time derivatives of p and q from the given Hamiltonian H
def calculate_acceleration(p, q, H):
    if p.volatile or q.volatile:
        p_nonv = Variable(p.data, requires_grad=True)
        q_nonv = Variable(q.data, requires_grad=True)
    else:
        p_nonv = p
        q_nonv = q
    hamilt = H(p_nonv, q_nonv)
    return - grad(hamilt.sum(), q_nonv, create_graph=True)[0]

def differentiate_hamiltonian(p, q, H):
    if p.volatile or q.volatile:
        p_nonv = Variable(p.data, requires_grad=True)
        q_nonv = Variable(q.data, requires_grad=True)
    else:
        p_nonv = p
        q_nonv = q
    hamilt = H(p_nonv, q_nonv)
    return -grad(hamilt.sum(), q_nonv, create_graph=True)[0], grad(hamilt.sum(), p_nonv, create_graph=True)[0]

def partial_H_partial_q(p, q, H):
    if p.volatile or q.volatile:
        p_nonv = Variable(p.data, requires_grad=True)
        q_nonv = Variable(q.data, requires_grad=True)
    else:
        p_nonv = p
        q_nonv = q
    hamilt = H(p_nonv, q_nonv)
    return grad(hamilt.sum(), q_nonv, create_graph=True)[0]

def partial_H_partial_p(p, q, H):
    if p.volatile or q.volatile:
        p_nonv = Variable(p.data, requires_grad=True)
        q_nonv = Variable(q.data, requires_grad=True)
    else:
        p_nonv = p
        q_nonv = q
    hamilt = H(p_nonv, q_nonv)
    return grad(hamilt.sum(), p_nonv, create_graph=True)[0]

## the symplectic "leapfrog" integrator
def leapfrog_old(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, p_0_is_var=False):
    # Assumes hamiltonian in potential form, hence p = q_dot
    if p_0_is_var:
        p_0_var = p_0
        q_0_var = q_0
    else:
        requires_grad = not volatile
        p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
        q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

    trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]),
            volatile=volatile)

    p = p_0_var
    q = q_0_var

    if is_Hamilt:
        accel = calculate_acceleration(p, q, Func)

        for i in range(T):
            trajectories[i, :, :p_0.shape[1]] = p
            trajectories[i, :, p_0.shape[1]:] = q

            p_half = p + accel * (dt / 2)
            q_next = q + p_half * dt

            accel_next = calculate_acceleration(p, q_next, Func)

            p_next = p_half + accel_next * (dt / 2)

            p = p_next
            q = q_next
            accel = accel_next

    else:
        state_0 = torch.cat((p, q), 1)
        time_drvt = Func(state_0)
        dpdt = time_drvt[:, :p_0.shape[1]]
        dqdt = time_drvt[:, p_0.shape[1]:]

        for i in range(T):
            trajectories[i, :, :p_0.shape[1]] = p
            trajectories[i, :, p_0.shape[1]:] = q

            p_half = p + dpdt * (dt / 2)
            q_next = q + dqdt * dt

            state = torch.cat((p, q_next), 1)
            time_drvt = Func(state)
            dpdt = time_drvt[:, :p_0.shape[1]]
            dqdt = time_drvt[:, p_0.shape[1]:]

            p_next = p_half + dpdt * (dt / 2)

            p = p_next
            q = q_next

    return trajectories

def leapfrog_slow(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, p_0_is_var=False):
    # Assumes hamiltonian in potential form, hence p = q_dot
    if p_0_is_var:
        p_0_var = p_0
        q_0_var = q_0
    else:
        requires_grad = not volatile
        p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
        q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

    trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]),
            volatile=volatile)

    p = p_0_var
    q = q_0_var

    if is_Hamilt:
        dpdt, dqdt = differentiate_hamiltonian(p, q, Func)
    else:
        state_0 = torch.cat((p, q), 1)
        time_drvt = Func(state_0)
        dpdt = time_drvt[:, :p_0.shape[1]]
        dqdt = time_drvt[:, p_0.shape[1]:]


    for i in range(T):
        trajectories[i, :, :p_0.shape[1]] = p
        trajectories[i, :, p_0.shape[1]:] = q

        p_half = p + dpdt * (dt / 2)

        if is_Hamilt:
            dpdt, dqdt = differentiate_hamiltonian(p_half, q, Func)
        else:
            state = torch.cat((p_half, q), 1)
            time_drvt = Func(state)
            dpdt = time_drvt[:, :p_0.shape[1]]
            dqdt = time_drvt[:, p_0.shape[1]:]

        q_next = q + dqdt * dt

        if is_Hamilt:
            dpdt, dqdt = differentiate_hamiltonian(p_half, q_next, Func)
        else:
            state = torch.cat((p_half, q_next), 1)
            time_drvt = Func(state)
            dpdt = time_drvt[:, :p_0.shape[1]]
            dqdt = time_drvt[:, p_0.shape[1]:]

        p_next = p_half + dpdt * (dt / 2)

        p = p_next
        q = q_next

    return trajectories

# def leapfrog(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, cuda=False):
#     # Assumes hamiltonian in potential form, hence p = q_dot
#     if p_0_is_var:
#         p_0_var = p_0
#         q_0_var = q_0
#     else:
#         requires_grad = not volatile
#         p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
#         q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

#     if cuda:
#         trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]).cuda(), volatile=volatile)
#     else:
#         trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]), volatile=volatile)

#     p = p_0_var
#     q = q_0_var

#     if is_Hamilt:
#         # dpdt, dqdt = differentiate_hamiltonian(p, q, Func)
#         dpdt = -partial_H_partial_q(p, q, Func)
#     else:
#         state_0 = torch.cat((p, q), 1)
#         time_drvt = Func(state_0)
#         dpdt = time_drvt[:, :p_0.shape[1]]

#     p_half = p + dpdt * (dt / 2)

#     for i in range(T):

#         # print ('inside lf')
#         # print ('p', p)
#         # print ('q', q)
#         trajectories[i, :, :p_0.shape[1]] = p
#         trajectories[i, :, p_0.shape[1]:] = q
#         # print ('traj', trajectories)

#         if is_Hamilt:
#             # dpdt, dqdt = differentiate_hamiltonian(p_half, q, Func)
#             dqdt = partial_H_partial_p(p_half, q, Func)
#         else:
#             state = torch.cat((p_half, q), 1)
#             time_drvt = Func(state)
#             dqdt = time_drvt[:, p_0.shape[1]:]

#         q_next = q + dqdt * dt

#         if is_Hamilt:
#             # dpdt, dqdt = differentiate_hamiltonian(p_half, q_next, Func)
#             dpdt = -partial_H_partial_q(p_half, q_next, Func)
#         else:
#             state = torch.cat((p_half, q_next), 1)
#             time_drvt = Func(state)
#             dpdt = time_drvt[:, :p_0.shape[1]]

#         p_half_next = p_half + dpdt * dt
#         p_next = (p_half + p_half_next) / 2

#         p = p_next
#         q = q_next
#         p_half = p_half_next

#     return trajectories


def symplectic_euler_p(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, p_0_is_var=False):
    # Assumes hamiltonian in potential form, hence p = q_dot
    if p_0_is_var:
        p_0_var = p_0
        q_0_var = q_0
    else:
        requires_grad = not volatile
        p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
        q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

    trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]),
            volatile=volatile)

    p = p_0_var
    q = q_0_var

    for i in range(T):
        trajectories[i, :, :p_0.shape[1]] = p
        trajectories[i, :, p_0.shape[1]:] = q

        if is_Hamilt:
            # dpdt, dqdt = differentiate_hamiltonian(p, q, Func)
            dpdt = -partial_H_partial_q(p, q, Func)
        else:
            state = torch.cat((p, q), 1)
            time_drvt = Func(state)
            dpdt = time_drvt[:, :p_0.shape[1]]
            dqdt = time_drvt[:, p_0.shape[1]:]

        p_next = p + dpdt * dt

        if is_Hamilt:
            # dpdt, dqdt = differentiate_hamiltonian(p_next, q, Func)
            dqdt = partial_H_partial_p(p_next, q, Func)
        else:
            state = torch.cat((p_next, q), 1)
            time_drvt = Func(state)
            dpdt = time_drvt[:, :p_0.shape[1]]
            dqdt = time_drvt[:, p_0.shape[1]:]

        q_next = q + dqdt * dt

        p = p_next
        q = q_next

    return trajectories


# def euler(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, p_0_is_var=False, cuda=False):
#     # Assumes hamiltonian in potential form, hence p = q_dot
#     if p_0_is_var:
#         p_0_var = p_0
#         q_0_var = q_0
#     else:
#         requires_grad = not volatile
#         p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
#         q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

#     if cuda:
#         trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]).cuda(), volatile=volatile)
#     else:
#         trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]), volatile=volatile)

#     p = p_0_var
#     q = q_0_var

#     if is_Hamilt:
#         accel = calculate_acceleration(p, q, Func)

#         for i in range(T):
#             trajectories[i, :, :p_0.shape[1]] = p
#             trajectories[i, :, p_0.shape[1]:] = q

#             p_next = p + accel * dt
#             q_next = q + p * dt

#             accel_next = calculate_acceleration(p_next, q_next, Func)

#             accel = accel_next

#             p = p_next
#             q = q_next

#     else:
#         state_0 = torch.cat((p, q), 1)
#         time_drvt = Func(state_0)
#         dpdt = time_drvt[:, :p_0.shape[1]]
#         dqdt = time_drvt[:, p_0.shape[1]:]

#         for i in range(T):
#             trajectories[i, :, :p_0.shape[1]] = p
#             trajectories[i, :, p_0.shape[1]:] = q

#             p_next = p + dpdt * dt
#             q_next = q + dqdt * dt

#             state = torch.cat((p_next, q_next), 1)
#             time_drvt = Func(state)
#             dpdt = time_drvt[:, :p_0.shape[1]]
#             dqdt = time_drvt[:, p_0.shape[1]:]

#             p = p_next
#             q = q_next

#     return trajectories



def plot_animation(n_v, n_h, simulated_traj, true_traj=None, plot_true_data = False):
    X = np.arange(n_h)
    Y = np.arange(n_v)
    X, Y = np.meshgrid(X, Y)
    T = simulated_traj.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.set_zlim(-10, 10)
    wframe = None

    if(plot_true_data==True):
        ax_2 = fig.add_subplot(212, projection='3d')
        ax_2.set_zlim(-10, 10)
        wframe_2 = None

    for k in range(T):
        if wframe:
            ax.collections.remove(wframe)
        Z = (simulated_traj[k, :]).view(n_v, n_h).data.numpy()

        wframe = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

        if(plot_true_data==True):
            if wframe_2:
                ax_2.collections.remove(wframe_2)
            Z_2 = (true_traj[k, :]).view(n_v, n_h).data.numpy()

            wframe_2 = ax_2.plot_wireframe(X, Y, Z_2, rstride=1, cstride=1)

        plt.pause(.01)

def plot_chain_animation(chain_length, simulated_traj, true_traj=None, plot_true_data = False):
    X = np.arange(chain_length)
    Y = np.arange(1)
    X, Y = np.meshgrid(X, Y)
    T = simulated_traj.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.set_zlim(-10, 10)
    wframe = None

    if(plot_true_data==True):
        ax_2 = fig.add_subplot(212, projection='3d')
        ax_2.set_zlim(-10, 10)
        wframe_2 = None

    for k in range(T):
        if wframe:
            ax.collections.remove(wframe)
        Z = (simulated_traj[k, :]).view(1, chain_length).data.numpy()

        wframe = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

        if(plot_true_data==True):
            if wframe_2:
                ax_2.collections.remove(wframe_2)
            Z_2 = (true_traj[k, :]).view(1, chain_length).data.numpy()

            wframe_2 = ax_2.plot_wireframe(X, Y, Z_2, rstride=1, cstride=1)

        plt.pause(.01)


def animation_demo():
    rows = 5
    cols = 6
    batch_size = 3
    p_0 = torch.zeros(batch_size, rows, cols)
    q_0 = torch.zeros(batch_size, rows, cols)
    q_0[:, 0, 0] = 10
    q_0_vec = q_0.view(batch_size, rows * cols)
    q_0_mat = q_0.view(batch_size, rows, cols)
    p_0_vec = p_0.view(batch_size, rows * cols)

    k = 5
    m = 1
    H = SprLatHamilt(rows, cols, k, m)
    T = 1000
    dt = 0.1
    trajectory = leapfrog(p_0_vec, q_0_vec, H, T, dt, volatile=True)

    plot_animation(rows, cols, trajectory[:, 0, rows * cols:])

def plot_1d_traj(simulated_traj, true_traj=None):
    t = np.arange(simulated_traj.shape[0])
    plt.plot(t, simulated_traj, t, true_traj)
    plt.pause(0.5)
    plt.show()
    plt.close()


## assumes that we know the formula for the Hamiltonian and only need to learn the parameters - mass and spring constant; not applicable to spring lattice or most real-life Hamiltonian systems
class QuadHamilt(nn.Module):
    def __init__(self):
        super(QuadHamilt, self).__init__()
        self.k = nn.Parameter(torch.rand(1))

    def forward(self, p, q):
        return 0.5 * p.pow(2).sum(dim=1) + self.k * 0.5 * q.pow(2).sum(dim=1)

## learns the Hamiltonian via MLP; assumes that the Hamiltonian is separable, that the kinetic energy is quadratic in p, and that we know the mass a priori (which we should change)
# class MLP_QuadKinetic_Hamilt(nn.Module):
#     def __init__(self, n_hidden, input_size):
#         super(MLP_QuadKinetic_Hamilt, self).__init__()
#         self.linear1 = nn.Linear(input_size, n_hidden)
#         self.linear2 = nn.Linear(n_hidden, 1)

#     def potential_energy(self, q):
#         h_pre = self.linear1(q)
#         h = h_pre.tanh_()
#         return self.linear2(h)

#     def forward(self, p, q):
#         # print ('q', q)
#         return 0.5 * p.pow(2).sum(dim=1, keepdim=True) + self.potential_energy(q)

# ## learns the Hamiltonian via MLP; assumes that the Hamiltonian is separable, but both the kinetic and the potential energy are unknown and learned via MLP
# class MLP_Separable_Hamilt(nn.Module):
#     def __init__(self, n_hidden, input_size):
#         super(MLP_Separable_Hamilt, self).__init__()
#         self.linear_K1 = nn.Linear(input_size, n_hidden)
#         self.linear_K2 = nn.Linear(n_hidden, 1)
#         self.linear_P1 = nn.Linear(input_size, n_hidden)
#         self.linear_P2 = nn.Linear(n_hidden, 1)
#         # if (know_p_0 == False):
#         #     self.p_0 = nn.Linear(input_size, 1)

#     def kinetic_energy(self, p):
#         h_pre = self.linear_K1(p)
#         h = h_pre.tanh_()
#         return self.linear_K2(h)

#     def potential_energy(self, q):
#         h_pre = self.linear_P1(q)
#         h = h_pre.tanh_()
#         return self.linear_P2(h)

#     def forward(self, p, q):
#         return self.kinetic_energy(p) + self.potential_energy(q)

# class MLP_General_Hamilt(nn.Module):
#     def __init__(self, n_hidden, input_size):
#         super(MLP_General_Hamilt, self).__init__()
#         self.linear1 = nn.Linear(input_size * 2, n_hidden)
#         self.linear1B = nn.Linear(n_hidden, n_hidden)
#         self.linear2 = nn.Linear(n_hidden, 1)

#         # if (know_p_0 == False):
#         #     self.p_0 = nn.Linear(input_size, 1)

#     def forward(self, p, q):
#         pq = torch.cat((p, q), 1)
#         h_pre = self.linear1(pq)
#         h = h_pre.tanh_()
#         h_pre_B = self.linear1B(h)
#         h_B = h_pre_B.tanh_()
#         return self.linear2(h_B)


class MLP1H(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(MLP1H, self).__init__()
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)

    def forward(self, x_input):
        hidden_pre = self.i2h(x_input)
        hidden = hidden_pre.tanh_()
        x_output = self.h2o(hidden)
        return x_output

class MLP_reg(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(MLP_reg, self).__init__()
        self.system_dim = n_output // 2
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)
        self.regularizer = 0

    def forward(self, x_input):
        hidden_pre = self.i2h(x_input)
        hidden = hidden_pre.tanh_()
        x_output = self.h2o(hidden)
        self.compute_regularizer(x_input, x_output)
        # batch_size = x_input.shape[0]
        # for i in range(self.system_dim):
        #     dpdotdp = grad(x_output[:, i].sum(), x_input[:, i], only_inputs = True, create_graph = True)
        #     print (dpdotdp.shape)
        #     dqdotdq = grad(x_output[:, i + dim].sum(), x_input[:, i + dim], only_inputs = True, create_graph = True)
        #     regularizer += norm(dpdotdp + dqdotdq)
        # self.regularizer = regularizer / batch_size
        return x_output

    def compute_regularizer(self, x_input, x_output):
        regularizer = 0
        batch_size = x_input.shape[0]
        for i in range(self.system_dim):
            dpdotdp = grad(x_output[:, i].sum(), x_input, only_inputs = True, create_graph = True)[0][:, i]
            # print (dpdotdp.shape)
            dqdotdq = grad(x_output[:, i + self.system_dim].sum(), x_input, only_inputs = True, create_graph = True)[0][:, i + self.system_dim]
            regularizer += torch.norm(dpdotdp + dqdotdq, p=1)
        self.regularizer = regularizer / batch_size


class MLP2H(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(MLP2H, self).__init__()
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2hB = nn.Linear(n_hidden, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)

    def forward(self, x_input):
        hidden_pre = self.i2h(x_input)
        hidden = hidden_pre.tanh_()
        hidden_B_pre = self.h2hB(hidden)
        hidden_B = hidden_B_pre.tanh_()
        x_output = self.h2o(hidden_B)
        return x_output

class MLP3H(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(MLP3H, self).__init__()
        self.i2hA = nn.Linear(n_input, n_hidden)
        self.hA2hB = nn.Linear(n_hidden, n_hidden)
        self.hB2hC = nn.Linear(n_hidden, n_hidden)
        self.hC2o = nn.Linear(n_hidden, n_output)

    def forward(self, x_input):
        hidden_A = self.i2hA(x_input).tanh_()
        hidden_B = self.hA2hB(hidden_A).tanh_()
        hidden_C = self.hB2hC(hidden_B).tanh_()
        x_output = self.hC2o(hidden_C)
        return x_output

class MLP_Separable(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(MLP_Separable, self).__init__()
        self.p2hq = nn.Linear(int(n_input / 2), n_hidden)
        self.q2hp = nn.Linear(int(n_input / 2), n_hidden)
        self.hq2oq = nn.Linear(n_hidden, int(n_output / 2))
        self.hp2op = nn.Linear(n_hidden, int(n_output / 2))
        self.system_dim = int(n_output / 2)
        # self.p2hq = nn.Linear(1, n_hidden)
        # self.q2hp = nn.Linear(1, n_hidden)
        # self.hq2oq = nn.Linear(n_hidden, 1)
        # self.hp2op = nn.Linear(n_hidden, 1)

    def forward(self, x_input):
        # hidden_pre_q = self.p2hq(x_input[:, :int(x_input.shape[1] / 2)])
        # hidden_pre_p = self.q2hp(x_input[:, int(x_input.shape[1] / 2):])
        hidden_pre_q = self.p2hq(x_input[:, :self.system_dim])
        hidden_pre_p = self.q2hp(x_input[:, self.system_dim:])
        hidden_q = hidden_pre_q.tanh_()
        hidden_p = hidden_pre_p.tanh_()
        x_output = torch.cat([self.hp2op(hidden_p), self.hq2oq(hidden_q)], dim=1)
        return x_output

class MLP_QuadKinetic_Hamilt(nn.Module):
    def __init__(self, n_hidden, input_size):
        super(MLP_QuadKinetic_Hamilt, self).__init__()
        self.linear1 = nn.Linear(input_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)

    def potential_energy(self, q):
        h_pre = self.linear1(q)
        h = h_pre.tanh_()
        return self.linear2(h)

    def forward(self, p, q):
        # print ('q', q)
        hamilt_val = 0.5 * p.pow(2).sum(dim=1, keepdim=True) + self.potential_energy(q)
        # print ('hamilt_val', hamilt_val)
        return hamilt_val

## learns the Hamiltonian via MLP; assumes that the Hamiltonian is separable, but both the kinetic and the potential energy are unknown and learned via MLP
# class MLP_Separable_Hamilt(nn.Module):
#     def __init__(self, n_hidden, input_size):
#         super(MLP_Separable_Hamilt, self).__init__()
#         self.linear_K1 = nn.Linear(input_size, n_hidden)
#         self.linear_K2 = nn.Linear(n_hidden, 1)
#         self.linear_P1 = nn.Linear(input_size, n_hidden)
#         self.linear_P2 = nn.Linear(n_hidden, 1)
#         # if (know_p_0 == False):
#         #     self.p_0 = nn.Linear(input_size, 1)

#     def kinetic_energy(self, p):
#         h_pre = self.linear_K1(p)
#         h = h_pre.tanh_()
#         return self.linear_K2(h)

#     def potential_energy(self, q):
#         h_pre = self.linear_P1(q)
#         h = h_pre.tanh_()
#         return self.linear_P2(h)

#     def forward(self, p, q):
#         return self.kinetic_energy(p) + self.potential_energy(q)

class MLP1H_Separable_Hamilt(nn.Module):
    def __init__(self, n_hidden, input_size):
        super(MLP1H_Separable_Hamilt, self).__init__()
        self.linear_K1 = nn.Linear(input_size, n_hidden)
        # self.linear_K1B = nn.Linear(n_hidden, n_hidden)
        self.linear_K2 = nn.Linear(n_hidden, 1)
        self.linear_P1 = nn.Linear(input_size, n_hidden)
        # self.linear_P1B = nn.Linear(n_hidden, n_hidden)
        self.linear_P2 = nn.Linear(n_hidden, 1)
        # if (know_p_0 == False):
        #     self.p_0 = nn.Linear(input_size, 1)

    def kinetic_energy(self, p):
        h_pre = self.linear_K1(p)
        h = h_pre.tanh_()
        # h_pre_B = self.linear_K1B(h)
        # h_B = h_pre_B.tanh_()
        return self.linear_K2(h)

    def potential_energy(self, q):
        h_pre = self.linear_P1(q)
        h = h_pre.tanh_()
        # h_pre_B = self.linear_P1B(h)
        # h_B = h_pre_B.tanh_()
        return self.linear_P2(h)

    def forward(self, p, q):
        return self.kinetic_energy(p) + self.potential_energy(q)

class MLP2H_Separable_Hamilt(nn.Module):
    def __init__(self, n_hidden, input_size):
        super(MLP2H_Separable_Hamilt, self).__init__()
        self.linear_K1 = nn.Linear(input_size, n_hidden)
        self.linear_K1B = nn.Linear(n_hidden, n_hidden)
        self.linear_K2 = nn.Linear(n_hidden, 1)
        self.linear_P1 = nn.Linear(input_size, n_hidden)
        self.linear_P1B = nn.Linear(n_hidden, n_hidden)
        self.linear_P2 = nn.Linear(n_hidden, 1)
        # if (know_p_0 == False):
        #     self.p_0 = nn.Linear(input_size, 1)

    def kinetic_energy(self, p):
        h_pre = self.linear_K1(p)
        h = h_pre.tanh_()
        h_pre_B = self.linear_K1B(h)
        h_B = h_pre_B.tanh_()
        return self.linear_K2(h_B)

    def potential_energy(self, q):
        h_pre = self.linear_P1(q)
        h = h_pre.tanh_()
        h_pre_B = self.linear_P1B(h)
        h_B = h_pre_B.tanh_()
        return self.linear_P2(h_B)

    def forward(self, p, q):
        return self.kinetic_energy(p) + self.potential_energy(q)

class MLP3H_Separable_Hamilt(nn.Module):
    def __init__(self, n_hidden, input_size):
        super(MLP3H_Separable_Hamilt, self).__init__()
        self.linear_K0 = nn.Linear(input_size, n_hidden)
        self.linear_K1 = nn.Linear(n_hidden, n_hidden)
        self.linear_K2 = nn.Linear(n_hidden, n_hidden)
        self.linear_K3 = nn.Linear(n_hidden, 1)
        self.linear_P0 = nn.Linear(input_size, n_hidden)
        self.linear_P1 = nn.Linear(n_hidden, n_hidden)
        self.linear_P2 = nn.Linear(n_hidden, n_hidden)
        self.linear_P3 = nn.Linear(n_hidden, 1)
        # if (know_p_0 == False):
        #     self.p_0 = nn.Linear(input_size, 1)

    def kinetic_energy(self, p):
        h = self.linear_K0(p).tanh_()
        h = self.linear_K1(h).tanh_()
        h = self.linear_K2(h).tanh_()
        return self.linear_K3(h)

    def potential_energy(self, q):
        h = self.linear_P0(q).tanh_()
        h = self.linear_P1(h).tanh_()
        h = self.linear_P2(h).tanh_()
        return self.linear_P3(h)

    def forward(self, p, q):
        return self.kinetic_energy(p) + self.potential_energy(q)

class MLP_General_Hamilt(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLP_General_Hamilt, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear1B = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)

        # if (know_p_0 == False):
        #     self.p_0 = nn.Linear(input_size, 1)

    def forward(self, p, q):
        pq = torch.cat((p, q), 1)
        h_pre = self.linear1(pq)
        h = h_pre.tanh_()
        h_pre_B = self.linear1B(h)
        h_B = h_pre_B.tanh_()
        return self.linear2(h_B)

class MLP_General_Hamilt_reg(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLP_General_Hamilt_reg, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear1B = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)
        self.system_dim = n_input // 2

        # if (know_p_0 == False):
        #     self.p_0 = nn.Linear(input_size, 1)

    def forward(self, p, q):
        pq = torch.cat((p, q), 1)
        h_pre = self.linear1(pq)
        h = h_pre.tanh_()
        h_pre_B = self.linear1B(h)
        h_B = h_pre_B.tanh_()
        output = self.linear2(h_B)
        self.compute_regularizer(output, p, q)
        return output

    def compute_regularizer(self, output, p, q):
        regularizer = 0
        pdot = - grad(output.sum(), q, only_inputs = True, create_graph = True)[0]
        qdot = grad(output.sum(), p, only_inputs = True, create_graph = True)[0]
        batch_size = p.shape[0]
        for i in range(self.system_dim):
            dpdotdp = grad(pdot[:, i].sum(), p, only_inputs = True, retain_graph = True)[0][:, i].data.numpy()
            # print (dpdotdp.shape)
            dqdotdq = grad(qdot[:, i].sum(), q, only_inputs = True, retain_graph = True)[0][:, i].data.numpy()
            regularizer += norm(dpdotdp + dqdotdq)
        self.regularizer = regularizer / batch_size



## regular RNN; one example of the type 3 algorithms in the writeup
class PlainRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PlainRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x_0, T, volatile=True):
        # x_0_var = Variable(x_0)
        batch_size = x_0.shape[0]
        # trajectory_simulated = Variable(torch.Tensor(T, batch_size, self.output_size))
        trajectories = torch.zeros((T, batch_size, self.output_size), requires_grad = not volatile).to(next(self.parameters()).device)
        hidden_init = torch.zeros(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)

        trajectories[0, :, :] = x_0
        x_input = x_0
        hidden = hidden_init
        # output, hidden = self.forward_step(x_0_var, hidden_init)
        # trajectory_simulated[1, :, :] = output
        # empty_input = Variable(torch.zeros(batch_size, self.input_size), requires_grad = False)
        for t in range(T-1):
            x_output, hidden = self.forward_step(x_input, hidden)
            trajectories[t+1, :, :] = x_output
            x_input = x_output
        return trajectories#, hidden

    def forward_step(self, x_input, hidden):
        #combined = torch.cat((input, hidden), 1)
        hidden_pre = self.i2h(x_input) + self.h2h(hidden)
        hidden = hidden_pre.tanh_()
        x_output = self.h2o(hidden)
        return x_output, hidden

## Multi-Layer Perceptron; We use it to output p_{t+1}, q_{t+1} as a (learned) function of only p_{t} and q_{t} at each update, another example of the type 3 algorithms in the writeup
class NaiveMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, cuda=False):
        super(NaiveMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.cuda = cuda

    def forward(self, x_0, T):
        #print ('T ', T)
        if (T == 0):
            if self.cuda:
                trajectory_simulated = Variable(torch.Tensor(0, batch_size, self.output_size).cuda())
            else:
                trajectory_simulated = Variable(torch.Tensor(0, batch_size, self.output_size))
            return trajectory_simulated
        x_0_var = Variable(x_0)
        batch_size = x_0.shape[0]
        trajectory_simulated = Variable(torch.Tensor(T, batch_size, self.output_size))
        #hidden_init = Variable(torch.zeros(batch_size, self.hidden_size))

        trajectory_simulated[0, :, :] = x_0_var
        if (T == 1):
            return trajectory_simulated

        x_input = x_0_var
        #hidden = hidden_init
        for t in range(T-1):
            x_output, hidden = self.forward_step(x_input)
            trajectory_simulated[t+1, :, :] = x_output
            x_input = x_output
        return trajectory_simulated#, hidden

    def forward_step(self, x_input):
        hidden_pre = self.i2h(x_input)
        hidden = hidden_pre.tanh_()
        x_output = self.h2o(hidden)
        return x_output, hidden

## LSTM; another example of the type 3 algorithms in the writeup
class myLSTM(nn.Module):
    def __init__(self, input_size, hco_size):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hco_size = hco_size

        self.i2gf = nn.Linear(input_size, hco_size)
        self.i2gi = nn.Linear(input_size, hco_size)
        self.i2go = nn.Linear(input_size, hco_size)
        self.i2cp = nn.Linear(input_size, hco_size)
        self.h2gf = nn.Linear(hco_size, hco_size)
        self.h2gi = nn.Linear(hco_size, hco_size)
        self.h2go = nn.Linear(hco_size, hco_size)
        self.h2cp = nn.Linear(hco_size, hco_size)

    def forward(self, x_0, T, volatile=True):
        batch_size = x_0.shape[0]
        trajectory_predicted = torch.zeros(T, batch_size, self.hco_size, requires_grad = not volatile).to(next(self.parameters()).device)
        hidden_init = torch.zeros(batch_size, self.hco_size, requires_grad = not volatile).to(next(self.parameters()).device)
        cell_init = torch.zeros(batch_size, self.hco_size, requires_grad = not volatile).to(next(self.parameters()).device)

        trajectory_predicted[0, :, :] = x_0
        x_input = x_0
        hidden = hidden_init
        cell = cell_init

        for t in range(T-1):
            cell, hidden = self.forward_step(x_input, hidden, cell)
            trajectory_predicted[t+1, :, :] = hidden
            x_input = hidden

        return trajectory_predicted#, hidden

    def forward_step(self, x_input, hidden, cell):
        gate_f = (self.h2gf(hidden) + self.i2gf(x_input)).sigmoid_()    # size: same as hidden/cell/output
        gate_i = (self.h2gi(hidden) + self.i2gi(x_input)).sigmoid_()    # size: same as hco
        cell_pre = (self.h2cp(hidden) + self.i2cp(x_input)).tanh_()     # size: same as hco
        cell = gate_f * cell + gate_i * cell_pre                        # size: same as hco
        gate_o = (self.h2go(hidden) + self.i2go(x_input)).sigmoid_()    # size: same as hco
        hidden = gate_o * cell.tanh_()

        return cell, hidden

## MLP used for learning the time derivatives of p and q as functions of p and q; corresponds to the type 2 algorithm in the writeup
class MLPTimeDrvt(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLPTimeDrvt, self).__init__()
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_input)

    def forward(self, x_input):
        hidden_pre = self.i2h(x_input)
        hidden = hidden_pre.tanh_()
        x_output = self.h2o(hidden)
        return x_output

class MLPTimeDrvt_Separable(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLPTimeDrvt_Separable, self).__init__()
        # self.p2hq = nn.Linear(int(n_input / 2), n_hidden)
        # self.q2hp = nn.Linear(int(n_input / 2), n_hidden)
        # self.hq2oq = nn.Linear(n_hidden, int(n_input / 2))
        # self.hp2op = nn.Linear(n_hidden, int(n_input / 2))
        self.p2hq = nn.Linear(1, n_hidden)
        self.q2hp = nn.Linear(1, n_hidden)
        self.hq2oq = nn.Linear(n_hidden, 1)
        self.hp2op = nn.Linear(n_hidden, 1)

    def forward(self, x_input):
        # hidden_pre_q = self.p2hq(x_input[:, :int(x_input.shape[1] / 2)])
        # hidden_pre_p = self.q2hp(x_input[:, int(x_input.shape[1] / 2):])
        hidden_pre_q = self.p2hq(x_input[:, :1])
        hidden_pre_p = self.q2hp(x_input[:, 1:])
        hidden_q = hidden_pre_q.tanh_()
        hidden_p = hidden_pre_p.tanh_()
        x_output = torch.cat([self.hp2op(hidden_p), self.hq2oq(hidden_q)], dim=1)
        return x_output

## Dynamic time warping loss, implemented in the way on Wikipedia
def dtw_loss(true_traj, pred_traj):
    def mse(v, w):
        return np.mean(np.power(v.numpy() - w.numpy(), 2))
    T = true_traj.shape[0]
    S = pred_traj.shape[0]
    # W = 0 / (T-1)
    W = 0
    arr = np.zeros([S, T])
    arr[0, 0] = mse(true_traj[0, :], pred_traj[0, :])
    for t in range(T - 1):
        arr[0, t+1] = np.inf
    for s in range(S - 1):
        arr[s+1, 0] = np.inf
    for s in range(S - 1):
        for t in range(T - 1):
            arr[s+1, t+1] = min(arr[s, t], arr[s, t+1], arr[s+1, t]) + mse(true_traj[t+1, :], pred_traj[s+1, :])
    min_loss = np.min(arr[:, T-1]) / T
    Tau = np.argmin(arr[:, T-1]) + 1
    return min_loss, Tau


## Also a loss function with dynamic time warping (implemented in my own way (a more complicated way...))
def loss_dp(true_traj, pred_traj):
    def mse(v, w):
        # return np.mean(np.power(v - w, 2))
        n = v.shape[0]
        sos = 0
        for i in range(n):
            element = v[i]-w[i]
            sos += pow(element, 2)
        sos = sos / n
        return sos
    # mse = mean_squared_error
    T = true_traj.shape[0]
    S = pred_traj.shape[0]
    # W = 0 / (T-1)
    W = 0
    arr = np.zeros([S, T])
    arr[0, 0] = mse(true_traj[0, :], pred_traj[0, :])
    for t in range(T - 1):
        arr[0, t+1] = np.inf
    for s in range(S - 1):
        arr[s+1, 0] = np.inf
    arr[1, 1] = arr[0, 0] + mse(true_traj[1, :], pred_traj[1, :])
    for t in range(T - 2):
        num_1 = arr[0, t] + mse(true_traj[t+1, :], (pred_traj[0, :] + pred_traj[1, :])/2) + mse(true_traj[t+2, :], pred_traj[1, :]) + W*2
        num_2 = arr[0, t+1] + mse(true_traj[t+2, :], pred_traj[1, :])
        arr[1, t+2] = min(num_1, num_2)
    for s in range(S - 2):
        # num_3 = arr[s, 0] + mse(true_traj[s+2, :], pred_traj[1, :]) + W
        num_3 = arr[s, 0] + mse(true_traj[1, :], pred_traj[s+2, :]) + W
        num_4 = arr[s+1, 0] + mse(true_traj[1, :], pred_traj[s+2, :])
        arr[s+2, 1] = min(num_3, num_4)
    for s in range(S - 2):
        for t in range(T - 2):
            num_1 = arr[s+1, t+1]
            num_2 = arr[s+1, t] + mse(true_traj[t+1, :], (pred_traj[s+1, :] + pred_traj[s+2, :])/2) + W*2
            num_3 = arr[s, t+1] + W
            arr[s+2, t+2] = min(num_1, num_2, num_3) + mse(true_traj[t+2, :], pred_traj[s+2, :])
    min_loss = np.min(arr[:, T-1]) / T
    Tau = np.argmin(arr[:, T-1]) + 1
    return min_loss, Tau

## A helper function, which calls one of the two dtw loss functions above
def time_elastic_loss(data, predicted, dtw=True):
    batch_size = data.shape[1]
    loss_arr = np.zeros(batch_size)
    for index in range(batch_size):
        true_traj = data[:, index, :]
        pred_traj = predicted[:, index, :]
        if dtw:
            loss_arr[index] = dtw_loss(true_traj, pred_traj)[0]
        else:
            loss_arr[index] = loss_dp(true_traj, pred_traj)[0]
    return loss_arr

def mse_loss(data, predicted):
    batch_size = data.shape[1]
    loss_arr = np.zeros(batch_size)
    for index in range(batch_size):
        true_traj = data[:, index, :]
        pred_traj = predicted[:, index, :]
        loss_arr[index] = np.mean(pow(true_traj.data.numpy() - pred_traj.data.numpy(), 2))
    return loss_arr

## tests the loss function
def testLossFunc():
    # true_traj = np.arange(5)
    # pred_traj = np.arange(5)
    k_true = 5
    m_true = 1
    # n_samples = 2000
    n_samples = 2
    T = 100
    dt = 0.1
    lr = 5 * 1e-3
    n_hidden=128
    n_v = 3
    n_h = 5
    data = create_spring_lattice_dataset(leap=True, n=1, T=T*2, dt=dt/2, k=k_true, m = m_true, n_v = n_v, n_h = n_h, coarsen=False)
    true_traj = data[np.arange(T) * 2, :, :]
    true_traj = true_traj[:, 0, :]
     # pred_traj = data[:, 0, :] + torch.randn(data.shape[0], data.shape[2])/10
    pred_traj = data[:, 0, :]
    # pred_traj[:T, :] = true_traj
    # for s in range(len(pred_traj)):
        # pred_traj[s, :] = 10000
    return loss_dp(true_traj, pred_traj)



## training

def train(data, system_type, method, T, batch_size, n_epochs, n_samples, dt, lr, n_hidden, n_v=1, n_h=1, chain_length=20, know_p_0=True, lf=True):

    dim = int(data.shape[2] / 2)

    if (method == 0):
        model = MLP_QuadKinetic_Hamilt(n_hidden, dim) #QuadHamilt()
    elif (method == 1):
        model = MLPTimeDrvt(2 * dim, n_hidden)
    elif (method == 2):
        model = NaiveMLP(2 * dim, n_hidden, 2 * dim)
    elif (method == 3):
        model = PlainRNN(2 * dim, n_hidden, 2 * dim)
    elif (method == 4):
        model = myLSTM(2 * dim, 2 * dim)
    elif (method == 5):
        model = MLP_Separable_Hamilt(n_hidden, dim)
    elif (method == 6):
        model = MLPTimeDrvt_Separable(2 * dim, n_hidden)
    elif (method == 7):
        model = MLP_General_Hamilt(n_hidden, dim)

    mse = nn.MSELoss()

    if (torch.cuda.is_available()):
        data = data.cuda()
        model = model.cuda()

    q_0 = data[0, :, dim:]
    if know_p_0:
        p_0 = data[0, :, :dim]
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        p_0 = torch.rand(q_0.shape[0], dim)
        p_0_var = Variable(p_0, requires_grad = True)
        q_0_var = Variable(q_0, requires_grad = True)
        params_lst = list(model.parameters())
        params_lst.append(p_0_var)
        opt = torch.optim.Adam(params_lst, lr=lr)

    for epoch in range(n_epochs):

        # max_t = int(round(((epoch + 1.) / n_epochs) * T))
        max_t = min(int(round((epoch / n_epochs) * T)) + 2, T)
        ## we use curriculum learning, which means that at the beginning of the training process we feed the algorithm with trajectories with shorter time lengths, and later on we gradually increase the length of the trajectories
        ## max_t incidates how long we want the trajectories of the training data to be at the current epoch

        if (max_t > 1):
            # perm = torch.randperm(n_samples)
            # perm = torch.arange(n_samples).type(torch.LongTensor)
            perm = torch.randperm(n_samples).numpy().tolist()
            data_permed = data[:, perm, :]
            if know_p_0:
                p_0_permed = p_0[perm, :]
                q_0_permed = q_0[perm, :]
            else:
                p_0_var_permed = p_0_var[perm, :]
                q_0_var_permed = q_0_var[perm, :]

            for i in range(0, n_samples, batch_size):
                opt.zero_grad()

                if i + batch_size > n_samples:
                    break
                batch = data_permed[:max_t, i:(i+batch_size), :]

                if know_p_0:
                    if (method == 0 or method == 5 or method == 7):
                        if lf:
                            trajectory_simulated = leapfrog(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, p_0_is_var = not know_p_0, cuda=torch.cuda.is_available())
                        else:
                            trajectory_simulated = euler(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, p_0_is_var = not know_p_0, cuda=torch.cuda.is_available())
                    elif (method == 1 or method == 6):
                        if lf:
                            trajectory_simulated = leapfrog(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, p_0_is_var = not know_p_0, cuda=torch.cuda.is_available())
                        else:
                            trajectory_simulated = euler(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, p_0_is_var = not know_p_0, cuda=torch.cuda.is_available())
                    else:
                        trajectory_simulated = model(batch[0, :, :], max_t * 2)
                else:
                    if (method == 0 or method == 5 or method == 7):
                        if lf:
                            trajectory_simulated = leapfrog(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, p_0_is_var = not know_p_0, cuda=torch.cuda.is_available())
                        else:
                            trajectory_simulated = euler(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, p_0_is_var = not know_p_0, cuda=torch.cuda.is_available())
                    elif (method == 1 or method == 6):
                        if lf:
                            trajectory_simulated = leapfrog(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, p_0_is_var = not know_p_0, cuda=torch.cuda.is_available())
                        else:
                            trajectory_simulated = euler(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, p_0_is_var = not know_p_0, cuda=torch.cuda.is_available())
                    else:
                        trajectory_simulated = model(torch.cat([p_0_var_permed[i:(i+batch_size), :].data, q_0_var_permed[i:(i+batch_size), :].data], 1), max_t)

                ## If we know both the true p trajectory and the true q trajectory:
                # error_total = mse(trajectory_simulated[:max_t, :, :], Variable(batch))
                # error_total.backward()
                ## If we only know the true q trajectory:
                error_total = mse(trajectory_simulated[:max_t, :, dim:], Variable(batch[:, :, dim:]))
                error_total.backward()

                # For RNN, clip the gradients with max norm 0.25
                if(method == 3):
                    torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

                opt.step()

                print(epoch, i, max_t, 'train', error_total.data[0])


    if know_p_0:
        return model
    else:
        return model, p_0_var

def train_new(data, system_type, method, T, batch_size, n_epochs, n_samples, dt, lr, n_hidden, chain_length=20, know_p_0=True, lf=True, regularization_constant=1, logger=None, device='cpu', val_data=None, n_val_samples=0, T_val=0, leapfrog_val=True):

    dim = int(data.shape[2] / 2)

    # if (method == 0):
    #     model = MLP_QuadKinetic_Hamilt(n_hidden, dim) #QuadHamilt()
    # elif (method == 1):
    #     model = MLPTimeDrvt(2 * dim, n_hidden)
    # elif (method == 2):
    #     model = NaiveMLP(2 * dim, n_hidden, 2 * dim)
    # elif (method == 3):
    #     model = PlainRNN(2 * dim, n_hidden, 2 * dim)
    # elif (method == 4):
    #     model = myLSTM(2 * dim, 2 * dim)
    # elif (method == 5):
    #     model = MLP_Separable_Hamilt(n_hidden, dim)
    # elif (method == 6):
    #     model = MLPTimeDrvt_Separable(2 * dim, n_hidden)
    # elif (method == 7):
    #     model = MLP_General_Hamilt(n_hidden, dim)

    if (method == 0):
        model = MLP_QuadKinetic_Hamilt(n_hidden, dim) #QuadHamilt()
    elif (method == 1) or (method == 21):
        model = MLP(2 * dim, n_hidden, 2 * dim)
    elif (method == 2):
        model = NaiveMLP(2 * dim, n_hidden, 2 * dim)
    elif (method == 3):
        model = PlainRNN(2 * dim, n_hidden, 2 * dim)
    elif (method == 4):
        model = myLSTM(2 * dim, 2 * dim)
    elif (method == 5) or (method == 25):
        model = MLP_Separable_Hamilt(n_hidden, dim)
    elif (method == 6):     # Only for pendulum
        model = MLP_Separable(2 * dim, n_hidden, 2 * dim)
    elif (method == 7) or (method == 27):
        model = MLP_General_Hamilt(2 * dim, n_hidden)
    elif (method == 8) or (method == 28):
        model = MLP_reg(2 * dim, n_hidden, 2 * dim)
    elif (method == 9):
        model = MLP_General_Hamilt_reg(2 * dim, n_hidden)
    elif (method == 10) or (method == 30):
        model = MLP_Separable(2 * dim, n_hidden, 2 * dim)

    mse = nn.MSELoss()

    # if (torch.cuda.is_available()):
    #     if (not ((method == 2) or (method == 3) or (method==4))):
    #         data = data.cuda()
    #         model = model.cuda()
    if (not ((method == 2) or (method == 3) or (method==4))):
        data = data.to(device)
        model = model.to(device)
        if (not val_data is None):
            val_data = val_data.to(device)

    q_0 = data[0, :, dim:]
    if know_p_0:
        p_0 = data[0, :, :dim]
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        p_0 = torch.rand(q_0.shape[0], dim)
        # p_0_var = Variable(p_0, requires_grad = True)
        # q_0_var = Variable(q_0, requires_grad = True)
        p_0_var = p_0.requires_grad_()
        q_0_var = q_0.requires_grad_()

        params_lst = list(model.parameters())
        params_lst.append(p_0_var)
        opt = torch.optim.Adam(params_lst, lr=lr)

    loss_record = []
    val_loss_record = []

    for epoch in tqdm(range(n_epochs)):

        # max_t = int(round(((epoch + 1.) / n_epochs) * T))
        # max_t = min(int(round((epoch / n_epochs) * T)) + 2, T)
        max_t = T
        ## we use curriculum learning, which means that at the beginning of the training process we feed the algorithm with trajectories with shorter time lengths, and later on we gradually increase the length of the trajectories
        ## max_t incidates how long we want the trajectories of the training data to be at the current epoch

        if (max_t > 1):
            # perm = torch.randperm(n_samples)
            # perm = torch.arange(n_samples).type(torch.LongTensor)
            perm = torch.randperm(n_samples).numpy().tolist()
            data_permed = data[:, perm, :]
            if know_p_0:
                p_0_permed = p_0[perm, :]
                q_0_permed = q_0[perm, :]
            else:
                p_0_var_permed = p_0_var[perm, :]
                q_0_var_permed = q_0_var[perm, :]

            loss_record_epoch = []

            for i in range(0, n_samples, batch_size):
                opt.zero_grad()

                if i + batch_size > n_samples:
                    break
                batch = data_permed[:max_t, i:(i+batch_size), :]

                # device='cuda:0'

                if know_p_0:
                    if (method == 0 or method == 5 or method == 7 or method == 9):
                        if lf:
                            trajectory_simulated = leapfrog(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, device=device)
                        else:
                            trajectory_simulated = euler(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, device=device)
                    elif (method == 1 or method == 6 or method == 8 or method == 10):
                        if lf:
                            trajectory_simulated = leapfrog(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, device=device)
                        else:
                            trajectory_simulated = euler(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, device=device)
                    else:
                        trajectory_simulated = model(batch[0, :, :], max_t * 2)
                else:
                    if (method == 0 or method == 5 or method == 7 or method == 9):
                        if lf:
                            trajectory_simulated = leapfrog(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, device=device)
                        else:
                            trajectory_simulated = euler(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, device=device)
                    elif (method == 1 or method == 6 or method == 8 or method == 10):
                        if lf:
                            trajectory_simulated = leapfrog(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, device=device)
                        else:
                            trajectory_simulated = euler(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, device=device)
                    else:
                        trajectory_simulated = model(torch.cat([p_0_var_permed[i:(i+batch_size), :].data, q_0_var_permed[i:(i+batch_size), :].data], 1), max_t)

                ## If we know both the true p trajectory and the true q trajectory:
                error_total = mse(trajectory_simulated[:max_t, :, :], Variable(batch))

                ## If we only know the true q trajectory:
                # error_total = mse(trajectory_simulated[:max_t, :, dim:], Variable(batch[:, :, dim:]))

                if (method % 20 == 8 or method % 20 == 9):
                    regularizer = model.regularizer * regularization_constant
                    # print ('regularizer', regularizer)
                    # error_total = error + regularizer
                    error_total += regularizer

                error_total.backward()

                # For RNN, clip the gradients with max norm 0.25
                if(method == 3):
                    torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

                opt.step()

                loss_record_epoch.append(error_total.item())

                # if (logger is None):
                #     # print(epoch, i, max_t, error_total.data[0])
                #     print(epoch, i, max_t, error_total.item())
                #     # "epoch %d train: p q seq loss %.6f image loss %.6f init p q prediction loss %.6f train loss %.6f" %
                #     # ((t,) + tuple(np.array(local_loss_record).mean(axis=0)))
                #     # print ("epoch %d i %d max_t %d loss %.6f" % (epoch, i, max_t, error_total.item()))
                # else:
                #     # logger.log(epoch, i, max_t, error_total.item())
                #     logger.log("epoch %d i %d max_t %d loss %.6f" % (epoch, i, max_t, error_total.item()))

            avg_loss_epoch = sum(loss_record_epoch) / len(loss_record_epoch)
            loss_record.append(avg_loss_epoch)



            # f = open(os.path.join(outdir, 'loss.pkl'), 'wb')
            # pickle.dump([np.array(loss_record), np.array(loss_record_te), t], f)
            # f.close()
            # loss_plot(os.path.join(outdir, 'loss.pkl'), outdir,name=['','','loss p','loss q'])

            if know_p_0:
                init_val_data = val_data[0, :, :]
                pred_val = predict(init_val_data, model=model, system_type=system_type, method=method, T_test=T_val, n_test_samples=n_val_samples, dt=dt, know_p_0 = True, lf=leapfrog_val, device=device)
                # val_error = mse(pred_val[:T_val, :, dim:], val_data[:T_val, :n_val_samples, dim:])
                val_error = mse(pred_val[:T_val, :, :], val_data[:T_val, :n_val_samples, :])
                print(epoch, i, max_t, 'test', val_error.item())
                val_loss_record.append(val_error.item())
            # else:


            if (logger is None):
                # print(epoch, i, max_t, error_total.data[0])
                print(epoch, i, max_t, 'trloss', error_total.item(), 'valoss', val_error.item())
                # "epoch %d train: p q seq loss %.6f image loss %.6f init p q prediction loss %.6f train loss %.6f" %
                # ((t,) + tuple(np.array(local_loss_record).mean(axis=0)))
                # print ("epoch %d i %d max_t %d loss %.6f" % (epoch, i, max_t, error_total.item()))
            else:
                # logger.log(epoch, i, max_t, error_total.item())
                logger.log("epoch %d max_t %d trloss %.6f valoss %.6f" % (epoch, max_t, avg_loss_epoch, val_error.item()))


    if know_p_0:
        return model, loss_record, val_loss_record
    else:
        return model, p_0_var, loss_record#, loss_record_val


def predict(init_state, model, system_type, method, T_test, n_test_samples, dt, chain_length=20, coarsen=False, know_p_0=True, p_0_pred=0, lf=True, device='cpu'):
    dim = int(init_state.shape[1] / 2)

    if (torch.cuda.is_available()):
        if (not ((method == 2) or (method == 3) or (method==4))):
            init_state = init_state.cuda()

    q_0 = init_state[:, dim:]
    if know_p_0:
        # p_0 = test_data[0, :, :dim]
        p_0 = init_state[:, :dim]
    else:
        p_0 = p_0_pred

    if (method == 0 or method == 5 or method == 7 or method == 9):
        if coarsen:
            coarsening_factor = 10
            if lf:
                fine_trajectory = leapfrog(p_0, q_0, model, T_test * coarsening_factor, dt / coarsening_factor, volatile=False, device=device)
            else:
                fine_trajectory = euler(p_0, q_0, model, T_test * coarsening_factor, dt / coarsening_factor, volatile=False, device=device)
            trajectory_predicted = fine_trajectory[np.arange(T_test) * coarsening_factor, :, :]
        else:
            if lf:
                trajectory_predicted = leapfrog(p_0, q_0, model, T_test, dt, volatile=False, device=device)
            else:
                trajectory_predicted = euler(p_0, q_0, model, T_test, dt, volatile=False, device=device)
    elif (method == 1 or method == 6 or method == 8 or method == 10):
        if lf:
            trajectory_predicted = leapfrog(p_0, q_0, model, T_test, dt, volatile=False, is_Hamilt=False, device=device)
        else:
            trajectory_predicted = euler(p_0, q_0, model, T_test, dt, volatile=False, is_Hamilt=False, device=device)
    else:
        # trajectory_predicted = model(test_data[0, :, :], T_test * 2)
        trajectory_predicted = model(torch.cat([p_0, q_0], 1), T_test)

    return trajectory_predicted


## testing
def test(test_data, model, system_type, method, T_test, n_test_samples, dt, n_v=1, n_h=1):
    if (system_type == 0):
        p_0 = test_data[0, :, 0].unsqueeze(1)
        q_0 = test_data[0, :, 1].unsqueeze(1)
    else:
        p_0 = test_data[0, :, :n_v * n_h]
        q_0 = test_data[0, :, n_v * n_h:]
    if (method == 0 or method == 5):
        trajectory_predicted = leapfrog(p_0, q_0, model, T_test * 2, dt, volatile=False)
    elif (method == 1 or method == 6):
        trajectory_predicted = leapfrog(p_0, q_0, model, T_test * 2, dt, volatile=False, is_Hamilt=False)
    else:
        trajectory_predicted = model(test_data[0, :, :], T_test * 2)

    mse = nn.MSELoss()
    test_error = mse(trajectory_predicted[:T_test, :, :], Variable(test_data))
    print ('MSE avg test error', test_error.data[0])
    mse_arr = mse_loss(trajectory_predicted[:T_test, :, :], Variable(test_data))
    np.save(experiment + str(T_test) + '_mse', mse_arr)
    print ('MSE arr', mse_arr)
    loss_arr = time_elastic_loss(test_data, trajectory_predicted.data, dtw=True)
    np.save(experiment + str(T_test) + '_dtw', loss_arr)
    print ('ElasticLoss avg test err', np.mean(loss_arr))

    if (system_type == 2):
        for i in range(0, n_test_samples, 1):
            plot_animation(n_v, n_h, trajectory_predicted[:T_test, i, n_v * n_h:], Variable(test_data[:, i, n_v * n_h:]), plot_true_data=True)
    else:
        for i in range(0, n_test_samples, 1):
            plot_1d_traj(trajectory_predicted[:T_test, i, 1].data.numpy(), test_data[:, i, 1].numpy())
    #         plot_animation(1, 1, trajectory_predicted[:T_test, i, 1].unsqueeze(1), Variable(test_data[:, i, 1].unsqueeze(1)), plot_true_data=True)


def main():

    # dataset_index = 'H'
    # noiseless_dataset_index = 'B'
    noiseless_dataset_index = 'D'
    # train_noise_level = 0.01
    train_noise_level = 0
    # dataset_index = noiseless_dataset_index + 'n' + str(train_noise_level)
    dataset_index = noiseless_dataset_index
    test_dataset_index = noiseless_dataset_index

    run_index = 'ni4_old_code'

    ## Parameters of the learning process
    batch_size = 32  ## batch size
    # batch_size = 1
    n_epochs = 50  ## number of epochs in training
    # n_epochs = 200
    # n_epochs = 100
    # n_epochs = 40
    # n_epochs = 3
    # T = 100  ## time length of training trajectories
    # T = 200
    # T = 4000
    T = 10

    # dt = 0.001  ## time step of numerical discretization
    dt = 0.1
    lr = 5 * 1e-3  ## learning rate
    # lr = 5 * 1e-2
    # lr = 1e-1
    n_hidden=256  ## number of hidden units in the MLP's or RNN's
    # n_hidden = 16
    # n_samples = 2000  ## number of training samples
    # n_samples = 512
    # n_samples = 448
    # n_samples = 384
    # n_samples = 944
    # n_samples = 32
    # n_samples = 1024
    # n_samples = 1024
    n_samples = 1000
    # n_test_samples = 30  ## number of testing samples
    n_test_samples = 32
    # T_test = 1000  ## time length of testing samples
    # T_test = 2000
    # T_test = 1000
    T_test = 50

    know_p_0 = True

    leapfrog_train = True
    leapfrog_test = True

    shorten = 0

    # T_short = 5
    # T_short = 2

    T_total = T + T_test

    regularization_constant = 1

    #### Training the models and making the predictions

    data_dir = './data/chain'
    model_dir = '../models/chain_' + str(dataset_index) + str(run_index)
    pred_dir = '../predictions/chain_' + str(dataset_index) + str(run_index)
    log_dir_together = '../logs/chain_' + str(dataset_index) + str(run_index)

    if (not os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    ## List of systems to be learned
    system_lst = ['chain']



    if (not os.path.isdir(model_dir)):
        os.mkdir(model_dir)
    if (not os.path.isdir(pred_dir)):
        os.mkdir(pred_dir)
    if (not os.path.isdir(log_dir_together)):
        os.mkdir(log_dir_together)


    # if (not path.exists(os.path.join(log_dir, 'trloss.log'))):
    #     os.mkdir(os.path.join(log_dir, 'trloss.log'))
    # if (not path.exists(os.path.join(log_dir, 'teloss.log'))):
    #     os.mkdir(os.path.join(log_dir, 'teloss.log'))

    print ('dataset', dataset_index)
    print ('run', run_index)

    for system_type in range(len(system_lst)):

        train_data_npy = np.load(data_dir + '/train_data_chain_' + dataset_index + '.npy')
        test_data_npy = np.load(data_dir + '/test_data_chain_' + test_dataset_index + '.npy')

        train_data = torch.from_numpy(train_data_npy[:T, :n_samples, :])
        test_data = torch.from_numpy(test_data_npy[:T_test, :n_test_samples, :])

        if (shorten == 1):
            train_data_shortened = torch.zeros(T_short, n_samples * int(T / T_short), train_data_npy.shape[2])
            # train_data_shortened = torch.zeros(T_short, n_samples * (T - T_short + 1), data_npy.shape[2])
            for i in range(int(T / T_short)):
            # for i in range(T - T_short + 1):
                train_data_shortened[:, i * n_samples : (i+1) * n_samples, :] = train_data[i * (T_short) : (i+1) * T_short, :n_samples, :]
                # train_data_shortened[:, i * n_samples : (i+1) * n_samples, :] = train_data[i : i + T_short, :n_samples, :]

            train_data = train_data_shortened
            T = T_short
            n_samples = train_data_shortened.shape[1]
        elif(shorten == 2):
            # train_data_shortened = torch.zeros(T_short, n_samples * int(T / T_short), data_npy.shape[2])
            train_data_shortened = torch.zeros(T_short, n_samples * (T - T_short + 1), train_data_npy.shape[2])
            # for i in range(int(T / T_short)):
            for i in range(T - T_short + 1):
                # train_data_shortened[:, i * n_samples : (i+1) * n_samples, :] = train_data[i * (T_short) : (i+1) * T_short, :n_samples, :]
                train_data_shortened[:, i * n_samples : (i+1) * n_samples, :] = train_data[i : i + T_short, :n_samples, :]

            train_data = train_data_shortened
            T = T_short
            n_samples = train_data_shortened.shape[1]

            print ('new num samples', n_samples)


        init_test_data = test_data[0, :, :]

        # for method in [8, 10, 5, 7, 1, 4, 3]:
        for method in [1, 5, 7]:
        ## 0: Leapfrog with MLP Hamiltonian
        ## 1: Leapfrog with MLP Time Derivative
        ## 2: MLP for predicting p_{t+1}, q_{t+1} from p_t, q_t
        ## 3: RNN
        ## 4: LSTM
        ## 5: Leapfrog with MLP for Separable Hamiltonian

            log_dir = '../logs/chain_' + str(dataset_index) + str(run_index) + '/m' + str(method)

            if (not os.path.isdir(log_dir)):
                os.mkdir(log_dir)

            logger0 = Logger(os.path.join(log_dir, 'trloss.log'), print_out=True)
            logger1 = Logger(os.path.join(log_dir, 'teloss.log'), print_out=True)

            start = time.time()

            ## Training the model
            if know_p_0:
                ## Case 1: p_0 is known
                # model = train(train_data, system_type=system_type, method=method, T=T, batch_size=batch_size, n_epochs=n_epochs, n_samples=n_samples, dt=dt, lr=lr, n_hidden=n_hidden, know_p_0=True, lf=leapfrog)
                model, loss_record, val_loss_record = train_new(train_data, system_type=system_type, method=method, T=T, batch_size=batch_size, n_epochs=n_epochs, n_samples=n_samples, dt=dt, lr=lr, n_hidden=n_hidden, know_p_0=True, lf=leapfrog_train, regularization_constant=regularization_constant, logger=logger0, device='cuda:0', val_data=test_data, n_val_samples=n_test_samples, T_val=T_test, leapfrog_val=leapfrog_test)

            else:
                ## Case 2: p_0 is not known and has to be learned for each trajectory
                model, p_0_var = train(train_data, system_type=system_type, method=method, T=T, batch_size=batch_size, n_epochs=n_epochs, n_samples=n_samples, dt=dt, lr=lr, n_hidden=n_hidden, know_p_0=False, lf=leapfrog_train)


            ## Saving the model
            if (torch.cuda.is_available()):
                torch.save(model.cpu(), model_dir + '/model_' + str(system_lst[system_type]) + '_' + str(method) + '_' + str(dataset_index) + '_' + str(run_index))
                model.cuda()
            else:
                torch.save(model, model_dir + '/model_' + str(system_lst[system_type]) + '_' + str(method) + '_' + str(dataset_index) + '_' + str(run_index))

            f = open(os.path.join(log_dir, 'loss.pkl'), 'wb')
            # pickle.dump([np.array(loss_record), np.array(loss_record_val), t], f)
            pickle.dump([np.array(loss_record), np.array(val_loss_record)], f)
            f.close()
            loss_plot(os.path.join(log_dir, 'loss.pkl'), log_dir, name=['','','loss p','loss q'])
            loss_plot_restricted(os.path.join(log_dir, 'loss.pkl'), log_dir, name=['','','loss p','loss q'])

            train_time = time.time() - start
            print ('training with method ' + str(method) + ' costs time ', train_time)

            ## Predicting the test trajectories
            if know_p_0:
                ## Case 1: p_0 of the trajectories is known
                traj_pred = predict(init_test_data, model=model, system_type=system_type, method=method, T_test=T_total, n_test_samples=n_test_samples, dt=dt, know_p_0 = True, lf=leapfrog_test, device='cuda:0')
            else:
                ## Case 2: p_0 of the trajectories is not known
                traj_pred = predict(init_test_data, model=model, system_type=system_type, method=method, T_test=T_total, n_test_samples=n_test_samples, dt=dt, know_p_0 = False, p_0_pred = p_0_var[:n_test_samples, :].data, lf=leapfrog_test, device='cuda:0')
                # traj_pred_cheat = predict(test_data, model=model, system_type=system_type, method=method, T_test=T_test, n_test_samples=n_test_samples, dt=dt, n_v = n_v, n_h = n_h, chain_length=chain_length, know_p_0 = True)

            pred_time = time.time() - start
            print ('making the predictions with method ' + str(method) + ' costs time ', pred_time)
            # print ('done making the prediction for' + str(system_lst[system_type]) + 'with method' + str(method))

            if (torch.cuda.is_available()):
                np.save(pred_dir + '/traj_pred_' + str(system_lst[system_type]) + '_' + str(method) + '_' + str(test_dataset_index) + '_' + str(run_index) + '.npy', traj_pred.cpu().data.numpy())
            else:
                ## Saving the predicted trajectories
                np.save(pred_dir + '/traj_pred_' + str(system_lst[system_type]) + '_' + str(method) + '_' + str(test_dataset_index) + '_' + str(run_index) + '.npy', traj_pred.data.numpy())

            print ('done saving the predicted trajectory')

if __name__ == "__main__":
    main()
