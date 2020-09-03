# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Earlier versions of this file were written by Zhengdao Chen, used with permission.

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from tqdm import tqdm


def leapfrog(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu', use_tqdm=False):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    if use_tqdm:
        range_of_for_loop = tqdm(range(T))
    else:
        range_of_for_loop = range(T)

    if is_Hamilt:
        hamilt = Func(p, q)
        dpdt = -grad(hamilt.sum(), q, create_graph=not volatile)[0]

        for i in range_of_for_loop:
            p_half = p + dpdt * (dt / 2)

            if volatile:
                trajectories[i, :, :p_0.shape[1]] = p.detach()
                trajectories[i, :, p_0.shape[1]:] = q.detach()
            else:
                trajectories[i, :, :p_0.shape[1]] = p
                trajectories[i, :, p_0.shape[1]:] = q

            hamilt = Func(p_half, q)
            dqdt = grad(hamilt.sum(), p, create_graph=not volatile)[0]

            q_next = q + dqdt * dt

            hamilt = Func(p_half, q_next)
            dpdt = -grad(hamilt.sum(), q_next, create_graph=not volatile)[0]

            p_next = p_half + dpdt * (dt / 2)

            p = p_next
            q = q_next

    else:
        dim = p_0.shape[1]
        time_drvt = Func(torch.cat((p, q), 1))
        dpdt = time_drvt[:, :dim]

        for i in range_of_for_loop:
            p_half = p + dpdt * (dt / 2)

            if volatile:
                trajectories[i, :, :dim] = p.detach()
                trajectories[i, :, dim:] = q.detach()
            else:
                trajectories[i, :, :dim] = p
                trajectories[i, :, dim:] = q

            time_drvt = Func(torch.cat((p_half, q), 1))
            dqdt = time_drvt[:, dim:]

            q_next = q + dqdt * dt

            time_drvt = Func(torch.cat((p_half, q_next), 1))
            dpdt = time_drvt[:, :dim]

            p_next = p_half + dpdt * (dt / 2)

            p = p_next
            q = q_next

    return trajectories


def euler(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu', use_tqdm=False):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    if use_tqdm:
        range_of_for_loop = tqdm(range(T))
    else:
        range_of_for_loop = range(T)

    if is_Hamilt:

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :p_0.shape[1]] = p.detach()
                trajectories[i, :, p_0.shape[1]:] = q.detach()
            else:
                trajectories[i, :, :p_0.shape[1]] = p
                trajectories[i, :, p_0.shape[1]:] = q

            hamilt = Func(p, q)
            dpdt = -grad(hamilt.sum(), q, create_graph=not volatile)[0]
            dqdt = grad(hamilt.sum(), p, create_graph=not volatile)[0]

            p_next = p + dpdt * dt
            q_next = q + dqdt * dt

            p = p_next
            q = q_next

    else:
        dim = p_0.shape[1]

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :dim] = p.detach()
                trajectories[i, :, dim:] = q.detach()
            else:
                trajectories[i, :, :dim] = p
                trajectories[i, :, dim:] = q

            time_drvt = Func(torch.cat((p, q), 1))
            dpdt = time_drvt[:, :dim]
            dqdt = time_drvt[:, dim:]

            p_next = p + dpdt * dt
            q_next = q + dqdt * dt

            p = p_next
            q = q_next

    return trajectories


def numerically_integrate(integrator, p_0, q_0, model, method, T, dt, volatile, device, coarsening_factor=1):
    if (coarsening_factor > 1):
        fine_trajectory = numerically_integrate(integrator, p_0, q_0, model, method, T * coarsening_factor, dt / coarsening_factor, volatile, device)
        trajectory_simulated = fine_trajectory[np.arange(T) * coarsening_factor, :, :]
        return trajectory_simulated
    if (method == 5):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrog(p_0, q_0, model, T, dt, volatile=volatile, device=device)
        elif (integrator == 'euler'):
            trajectory_simulated = euler(p_0, q_0, model, T, dt, volatile=volatile, device=device)
    elif (method == 1):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrog(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
        elif (integrator == 'euler'):
            trajectory_simulated = euler(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
    else:
        trajectory_simulated = model(torch.cat([p_0, q_0], dim=1), T)
    return trajectory_simulated


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


class MLP1H_Separable_Hamilt(nn.Module):
    def __init__(self, n_hidden, input_size):
        super(MLP1H_Separable_Hamilt, self).__init__()
        self.linear_K1 = nn.Linear(input_size, n_hidden)
        self.linear_K2 = nn.Linear(n_hidden, 1)
        self.linear_P1 = nn.Linear(input_size, n_hidden)
        self.linear_P2 = nn.Linear(n_hidden, 1)

    def kinetic_energy(self, p):
        h_pre = self.linear_K1(p)
        h = h_pre.tanh_()
        return self.linear_K2(h)

    def potential_energy(self, q):
        h_pre = self.linear_P1(q)
        h = h_pre.tanh_()
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

    def forward(self, p, q):
        pq = torch.cat((p, q), 1)
        h_pre = self.linear1(pq)
        h = h_pre.tanh_()
        h_pre_B = self.linear1B(h)
        h_B = h_pre_B.tanh_()
        return self.linear2(h_B)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x_0, T, volatile=True):
        batch_size = x_0.shape[0]
        trajectories = torch.empty((T, batch_size, self.output_size), requires_grad = not volatile).to(next(self.parameters()).device)
        hidden_init = torch.empty(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)

        trajectories[0, :, :] = x_0
        x_input = x_0
        hidden = hidden_init
        for t in range(T-1):
            x_output, hidden = self.forward_step(x_input, hidden)
            trajectories[t+1, :, :] = x_output
            x_input = x_output
        return trajectories

    def forward_step(self, x_input, hidden):
        hidden_pre = self.i2h(x_input) + self.h2h(hidden)
        hidden = hidden_pre.tanh_()
        x_output = self.h2o(hidden)
        return x_output, hidden


class LSTM(nn.Module):
    def __init__(self, input_size, hco_size):
        super(LSTM, self).__init__()
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

        return trajectory_predicted

    def forward_step(self, x_input, hidden, cell):
        gate_f = (self.h2gf(hidden) + self.i2gf(x_input)).sigmoid_()    # size: same as hidden/cell/output
        gate_i = (self.h2gi(hidden) + self.i2gi(x_input)).sigmoid_()    # size: same as hco
        cell_pre = (self.h2cp(hidden) + self.i2cp(x_input)).tanh_()     # size: same as hco
        cell = gate_f * cell + gate_i * cell_pre                        # size: same as hco
        gate_o = (self.h2go(hidden) + self.i2go(x_input)).sigmoid_()    # size: same as hco
        hidden = gate_o * cell.tanh_()

        return cell, hidden


## For initial state optimization
class PyTorchObjective(object):

    def __init__(self, true_data, model, dim, integrator, T, dt, device, method, batch_size=32, z_0_old=None):
        self.mse = nn.MSELoss()
        self.true_data = true_data.requires_grad_()
        self.loss = 0
        self.cached_x = None
        self.model = model
        self.dim = dim
        self.integrator = integrator
        self.T = T
        self.dt = dt
        self.device = device
        self.method = method
        self.batch_size = batch_size
        self.noisy_z_0 = true_data[0, :].to('cpu').detach().numpy()
        if (z_0_old is None):
            self.z_0_old = true_data[0, :].to('cpu').detach().numpy()
        else:
            self.z_0_old = z_0_old


    def fun(self, z_0_npy):
        z_0 = torch.from_numpy(z_0_npy).type(torch.FloatTensor).unsqueeze(0)
        z_0.requires_grad_()
        z_0_device = z_0.to(self.device)
        p_0 = z_0_device[:, :self.dim]
        q_0 = z_0_device[:, self.dim:]
        if (self.integrator == 'NA'):
            trajectory_simulated = model(torch.cat([p_0, q_0], 1), self.T)
        else:
            trajectory_simulated = numerically_integrate(integrator=self.integrator, p_0=p_0, q_0=q_0, model=self.model, method=self.method, T=self.T, dt=self.dt, volatile=False, device=self.device)
        loss = self.mse(trajectory_simulated, self.true_data)
        self.loss = loss
        return loss.item()

    def jac(self, z_0_npy):
        z_0 = torch.from_numpy(z_0_npy).type(torch.FloatTensor).unsqueeze(0)
        z_0.requires_grad_()
        z_0_device = z_0.to(self.device)
        p_0 = z_0_device[:, :self.dim]
        q_0 = z_0_device[:, self.dim:]
        trajectory_simulated = numerically_integrate(integrator=self.integrator, p_0=p_0, q_0=q_0, model=self.model, method=self.method, T=self.T, dt=self.dt, volatile=False, device=self.device)
        loss = self.mse(trajectory_simulated, self.true_data)
        loss.backward()
        grad = z_0.grad.type(torch.DoubleTensor).numpy()
        return grad

    def fun_batch(self, z_0_unrolled_npy):
        z_0 = torch.from_numpy(z_0_unrolled_npy).type(torch.FloatTensor).view(-1, self.dim * 2)
        z_0.requires_grad_()
        z_0_device = z_0.to(self.device)
        p_0 = z_0_device[:, :self.dim]
        q_0 = z_0_device[:, self.dim:]
        trajectory_simulated = numerically_integrate(integrator=self.integrator, p_0=p_0, q_0=q_0, model=self.model, method=self.method, T=self.T, dt=self.dt, volatile=True, device=self.device)
        loss = self.mse(trajectory_simulated, self.true_data)
        self.loss = loss
        return loss.item()

    def jac_batch(self, z_0_unrolled_npy):
        z_0 = torch.from_numpy(z_0_unrolled_npy).type(torch.FloatTensor).view(-1, self.dim * 2)
        z_0.requires_grad_()
        z_0_device = z_0.to(self.device)
        p_0 = z_0_device[:, :self.dim]
        q_0 = z_0_device[:, self.dim:]
        trajectory_simulated = numerically_integrate(integrator=self.integrator, p_0=p_0, q_0=q_0, model=self.model, method=self.method, T=self.T, dt=self.dt, volatile=False, device=self.device)
        loss = self.mse(trajectory_simulated, self.true_data)
        loss.backward()
        grad = z_0.grad.type(torch.DoubleTensor).view(z_0_unrolled_npy.shape[0]).numpy()
        return grad


