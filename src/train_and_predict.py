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

# from generate_datasets import calculate_acceleration, leapfrog

def differentiate_hamiltonian(p, q, H):
    if p.volatile or q.volatile:
        p_nonv = Variable(p.data, requires_grad=True)
        q_nonv = Variable(q.data, requires_grad=True)
    else:
        p_nonv = p
        q_nonv = q
    hamilt = H(p_nonv, q_nonv)
    return -grad(hamilt.sum(), q_nonv, create_graph=True, allow_unused=True)[0], grad(hamilt.sum(), p_nonv, create_graph=True, allow_unused=True)[0]

## calculates the time derivatives of p and q from the given Hamiltonian H
def calculate_acceleration(p, q, H):
    if p.volatile or q.volatile:
        p_nonv = Variable(p.data, requires_grad=True)
        q_nonv = Variable(q.data, requires_grad=True)
    else:
        p_nonv = p
        q_nonv = q
    hamilt = H(p_nonv, q_nonv)
    # print ('hamilt', hamilt)
    return - grad(hamilt.sum(), q_nonv, create_graph=True, allow_unused=True)[0]

def partial_H_partial_q(p, q, H):
    p.requires_grad_()
    q.requires_grad_()
    hamilt = H(p, q)
    return grad(hamilt.sum(), q, create_graph=True)[0]

def partial_H_partial_p(p, q, H):
    p.requires_grad_()
    q.requires_grad_()
    hamilt = H(p, q)
    return grad(hamilt.sum(), p, create_graph=True)[0]

## the symplectic "leapfrog" integrator
# def leapfrog(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu'):
#     # Assumes hamiltonian in potential form, hence p = q_dot
#     # if p_0_is_var:
#     #     p_0_var = p_0
#     #     q_0_var = q_0
#     # else:
#     #     requires_grad = not volatile
#     #     p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
#     #     q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

    
#     # trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]),
#     #         volatile=volatile)
#     trajectories = torch.zeros((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad = not volatile).to(device)
    
#     # p = p_0_var
#     # q = q_0_var

#     p = p_0
#     q = q_0
#     p.requires_grad_()
#     q.requires_grad_()
    
#     if is_Hamilt:
#         accel = calculate_acceleration(p, q, Func)

#         for i in tqdm(range(T)):
#         # for i in range(T):
#             trajectories[i, :, :p_0.shape[1]] = p
#             trajectories[i, :, p_0.shape[1]:] = q

#             p_half = p + accel * (dt / 2)
#             q_next = q + p_half * dt

#             accel_next = calculate_acceleration(p, q_next, Func)

#             # print (p)
#             # print (torch.max(torch.abs(accel_next)).item())
#             # break

#             p_next = p_half + accel_next * (dt / 2)

#             p = p_next
#             q = q_next
#             accel = accel_next

#     else:
#         state_0 = torch.cat((p, q), 1)
#         time_drvt = Func(state_0)
#         dpdt = time_drvt[:, :p_0.shape[1]]
#         dqdt = time_drvt[:, p_0.shape[1]:]

#         for i in range(T):
#             trajectories[i, :, :p_0.shape[1]] = p
#             trajectories[i, :, p_0.shape[1]:] = q

#             p_half = p + dpdt * (dt / 2)
#             q_next = q + dqdt * dt

#             state = torch.cat((p, q_next), 1)
#             time_drvt = Func(state)
#             dpdt = time_drvt[:, :p_0.shape[1]]
#             dqdt = time_drvt[:, p_0.shape[1]:]

#             p_next = p_half + dpdt * (dt / 2)

#             p = p_next
#             q = q_next

#     return trajectories


def leapfrogG(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu', use_tqdm=False):

    trajectories = torch.zeros((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad = not volatile).to(device)

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
                # p = p.detach().requires_grad_()
                # q = q.detach().requires_grad_()
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
        # dqdt = time_drvt[:, p_0.shape[1]:]

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



def leapfrog_stack(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu'):
    # Assumes hamiltonian in potential form, hence p = q_dot
    # if p_0_is_var:
    #     p_0_var = p_0
    #     q_0_var = q_0
    # else:
    #     requires_grad = not volatile
    #     p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
    #     q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

    
    # trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]),
    #         volatile=volatile)
    # trajectories = torch.zeros((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad = not volatile).to(device)
    frames = []
    
    # p = p_0_var
    # q = q_0_var

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()
    
    if is_Hamilt:
        accel = calculate_acceleration(p, q, Func)

        # for i in tqdm(range(T)):
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

# def leapfrog2(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu'):
#     # Assumes hamiltonian in potential form, hence p = q_dot
#     # if p_0_is_var:
#     #     p_0_var = p_0
#     #     q_0_var = q_0
#     # else:
#     #     requires_grad = not volatile
#     #     p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
#     #     q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

#     # trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]),
#     #         volatile=volatile)

#     trajectories = torch.zeros((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad = not volatile).to(device)

#     # p = p_0_var
#     # q = q_0_var

#     p = p_0
#     q = q_0
#     p.requires_grad_()
#     q.requires_grad_()

#     if is_Hamilt:
#         dpdt, dqdt = differentiate_hamiltonian(p, q, Func)
#     else:
#         state_0 = torch.cat((p, q), 1)
#         time_drvt = Func(state_0)
#         dpdt = time_drvt[:, :p_0.shape[1]]
#         dqdt = time_drvt[:, p_0.shape[1]:]


#     for i in range(T):
#         trajectories[i, :, :p_0.shape[1]] = p
#         trajectories[i, :, p_0.shape[1]:] = q

#         p_half = p + dpdt * (dt / 2)
#         q_next = q + dqdt * dt

#         if is_Hamilt:
#             dpdt, dqdt = differentiate_hamiltonian(p, q_next, Func)
#         else:
#             state = torch.cat((p, q_next), 1)
#             time_drvt = Func(state)
#             dpdt = time_drvt[:, :p_0.shape[1]]
#             dqdt = time_drvt[:, p_0.shape[1]:]

#         p_next = p_half + dpdt * (dt / 2)

#         p = p_next
#         q = q_next

#     return trajectories

# def euler(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu'):
#     # Assumes hamiltonian in potential form, hence p = q_dot
#     # if p_0_is_var:
#     #     p_0_var = p_0
#     #     q_0_var = q_0
#     # else:
#     #     requires_grad = not volatile
#     #     p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
#     #     q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

#     # if cuda:
#     #     trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]).cuda(), volatile=volatile)
#     # else:
#     #     trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]), volatile=volatile)

#     trajectories = torch.zeros((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad = not volatile).to(device)
#     # trajectories = torch.zeros((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad = False).to(device)

#     # p = p_0_var
#     # q = q_0_var

#     p = p_0
#     q = q_0
#     p.requires_grad_()
#     q.requires_grad_()

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


def eulerG(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu', use_tqdm=False):

    trajectories = torch.zeros((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad = not volatile).to(device)

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


def euler_stack(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu'):
    # Assumes hamiltonian in potential form, hence p = q_dot
    # if p_0_is_var:
    #     p_0_var = p_0
    #     q_0_var = q_0
    # else:
    #     requires_grad = not volatile
    #     p_0_var = Variable(p_0, volatile=volatile, requires_grad=requires_grad)
    #     q_0_var = Variable(q_0, volatile=volatile, requires_grad=requires_grad)

    # if cuda:
    #     trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]).cuda(), volatile=volatile)
    # else:
    #     trajectories = Variable(torch.Tensor(T, p_0.shape[0], 2 * p_0.shape[1]), volatile=volatile)

    # trajectories = torch.zeros((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad = not volatile).to(device)
    # trajectories = torch.zeros((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad = False).to(device)
    frames = []

    # p = p_0_var
    # q = q_0_var

    p = p_0
    q = q_0
    # p.requires_grad_()
    # q.requires_grad_()

    if is_Hamilt:
        accel = calculate_acceleration(p, q, Func)

        for i in range(T):
            # trajectories[i, :, :p_0.shape[1]] = p
            # trajectories[i, :, p_0.shape[1]:] = q
            frames.append(torch.cat([p, q], dim=1))

            p_next = p + accel * dt
            q_next = q + p * dt

            accel_next = calculate_acceleration(p_next, q_next, Func)

            accel = accel_next

            p = p_next
            q = q_next

    else:
        state_0 = torch.cat((p, q), dim=1)
        time_drvt = Func(state_0)
        dpdt = time_drvt[:, :p_0.shape[1]]
        dqdt = time_drvt[:, p_0.shape[1]:]

        for i in range(T):
            # trajectories[i, :, :p_0.shape[1]] = p
            # trajectories[i, :, p_0.shape[1]:] = q
            frames.append(torch.cat([p, q], dim=1))

            p_next = p + dpdt * dt
            q_next = q + dqdt * dt

            state = torch.cat((p_next, q_next), 1)
            time_drvt = Func(state)
            dpdt = time_drvt[:, :p_0.shape[1]]
            dqdt = time_drvt[:, p_0.shape[1]:]

            p = p_next
            q = q_next

    trajectories = torch.stack(frames, dim=0)

    # print (trajectories.size())

    return trajectories

def numerically_integrate(integrator, p_0, q_0, model, method, T, dt, volatile, device, coarsening_factor=1):
    if (coarsening_factor > 1):
        fine_trajectory = numerically_integrate(integrator, p_0, q_0, model, method, T * coarsening_factor, dt / coarsening_factor, volatile, device)
        trajectory_simulated = fine_trajectory[np.arange(T) * coarsening_factor, :, :]
        return trajectory_simulated
    if (method == 0 or method == 5 or method == 7 or method == 9):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrogG(p_0, q_0, model, T, dt, volatile=volatile, device=device)
        elif (integrator == 'euler'):
            # trajectory_simulated = euler_stack(p_0, q_0, model, T, dt, volatile=volatile, device=device)
            trajectory_simulated = eulerG(p_0, q_0, model, T, dt, volatile=volatile, device=device)
    elif (method == 1 or method == 6 or method == 8 or method == 10):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrogG(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
        elif (integrator == 'euler'):
            # trajectory_simulated = euler_stack(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
            trajectory_simulated = eulerG(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
    else:
        trajectory_simulated = model(torch.cat([p_0, q_0], dim=1), T)
    return trajectory_simulated

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
        return 0.5 * p.pow(2).sum(dim=1, keepdim=True) + self.potential_energy(q)

## learns the Hamiltonian via MLP; assumes that the Hamiltonian is separable, but both the kinetic and the potential energy are unknown and learned via MLP 
class MLP_Separable_Hamilt(nn.Module):
    def __init__(self, n_hidden, input_size):
        super(MLP_Separable_Hamilt, self).__init__()
        self.linear_K1 = nn.Linear(input_size, n_hidden)
        self.linear_K2 = nn.Linear(n_hidden, 1)
        self.linear_P1 = nn.Linear(input_size, n_hidden)
        self.linear_P2 = nn.Linear(n_hidden, 1)
        # if (know_p_0 == False):
        #     self.p_0 = nn.Linear(input_size, 1)

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
        
    def forward(self, x_0, T):
        x_0_var = Variable(x_0)
        batch_size = x_0.shape[0]
        trajectory_simulated = Variable(torch.Tensor(T, batch_size, self.output_size))
        hidden_init = Variable(torch.zeros(batch_size, self.hidden_size))

        trajectory_simulated[0, :, :] = x_0_var
        x_input = x_0_var
        hidden = hidden_init
        # output, hidden = self.forward_step(x_0_var, hidden_init)
        # trajectory_simulated[1, :, :] = output
        # empty_input = Variable(torch.zeros(batch_size, self.input_size), requires_grad = False)
        for t in range(T-1):
            x_output, hidden = self.forward_step(x_input, hidden)
            trajectory_simulated[t+1, :, :] = x_output
            x_input = x_output
        return trajectory_simulated#, hidden

    def forward_step(self, x_input, hidden):
        #combined = torch.cat((input, hidden), 1)
        hidden_pre = self.i2h(x_input) + self.h2h(hidden)
        hidden = hidden_pre.tanh_()
        x_output = self.h2o(hidden)
        return x_output, hidden

## Multi-Layer Perceptron; We use it to output p_{t+1}, q_{t+1} as a (learned) function of only p_{t} and q_{t} at each update, another example of the type 3 algorithms in the writeup
class NaiveMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NaiveMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
    def forward(self, x_0, T):
        #print ('T ', T)
        if (T == 0):
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
        
    def forward(self, x_0, T):
        x_0_var = Variable(x_0)
        batch_size = x_0.shape[0]
        trajectory_predicted = Variable(torch.Tensor(T, batch_size, self.hco_size))
        hidden_init = Variable(torch.zeros(batch_size, self.hco_size))
        cell_init = Variable(torch.zeros(batch_size, self.hco_size))

        trajectory_predicted[0, :, :] = x_0_var
        x_input = x_0_var
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

def train(data, system_type, method, T, batch_size, n_epochs, n_samples, dt, lr, n_hidden, n_v=1, n_h=1, chain_length=20, know_p_0=True):

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
    
    mse = nn.MSELoss()

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

        max_t = int(round(((epoch + 1.) / n_epochs) * T))
        ## we use curriculum learning, which means that at the beginning of the training process we feed the algorithm with trajectories with shorter time lengths, and later on we gradually increase the length of the trajectories
        ## max_t incidates how long we want the trajectories of the training data to be at the current epoch

        if (max_t > 1):
            perm = torch.randperm(n_samples)
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
                    if (method == 0 or method == 5):
                        trajectory_simulated = leapfrog(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, p_0_is_var = not know_p_0)
                    elif (method == 1 or method == 6):
                        trajectory_simulated = leapfrog(p_0_permed[i:(i+batch_size), :], q_0_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, p_0_is_var = not know_p_0)
                    else:
                        trajectory_simulated = model(batch[0, :, :], max_t * 2)
                else:
                    if (method == 0 or method == 5):
                        trajectory_simulated = leapfrog(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, p_0_is_var = not know_p_0)
                    elif (method == 1 or method == 6):
                        trajectory_simulated = leapfrog(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, p_0_is_var = not know_p_0)
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
                
                print(epoch, i, max_t, error_total.data[0])

    if know_p_0:
        return model
    else:
        return model, p_0_var


def predict(init_state, model, system_type, method, T_test, n_test_samples, dt, n_v=1, n_h=1, chain_length=20, coarsen=False, know_p_0=True, p_0_pred=0):
    dim = int(init_state.shape[1] / 2)
    q_0 = init_state[:, dim:]
    if know_p_0:
        # p_0 = test_data[0, :, :dim]
        p_0 = init_state[:, :dim]
    else:
        p_0 = p_0_pred

    if (method == 0 or method == 5):
        if coarsen:
            coarsening_factor = 10
            fine_trajectory = leapfrog(p_0, q_0, model, T_test * coarsening_factor * 2, dt / coarsening_factor, volatile=False)
            trajectory_predicted = fine_trajectory[np.arange(T_test) * coarsening_factor, :, :]
        else:
            trajectory_predicted = leapfrog(p_0, q_0, model, T_test * 2, dt, volatile=False)
    elif (method == 1 or method == 6):
        trajectory_predicted = leapfrog(p_0, q_0, model, T_test * 2, dt, volatile=False, is_Hamilt=False)
    else:
        # trajectory_predicted = model(test_data[0, :, :], T_test * 2)
        trajectory_predicted = model(torch.cat([p_0, q_0], 1), T_test * 2)

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
    dataset_index = 'A'
    run_index = '1'

    ## Parameters of the systems
    ## Size of the spring lattice
    n_v = 3  ## vertical size of the spring lattice
    n_h = 5  ## horizontal size of the spring lattice

    ## Size of the spring chain
    chain_length=20

    ## Parameters of the learning process
    batch_size = 32  ## batch size
    n_epochs = 50  ## number of epochs in training
    # n_epochs = 3
    T = 100  ## time length of training trajectories
    dt = 0.1  ## time step of numerical discretization
    lr = 5 * 1e-3  ## learning rate
    # lr = 1e-1
    n_hidden=128  ## number of hidden units in the MLP's or RNN's
    n_samples = 2000  ## number of training samples
    # n_samples = 32  
    n_test_samples = 30  ## number of testing samples
    T_test = 1000  ## time length of testing samples
    know_p_0 = False

    #### Training the models and making the predictions

    ## List of systems to be learned
    system_lst = ['pend', '3by5fb', 'sc20']
    # system_lst = ['3by5fb']
    ## 'pend': pendulum
    ## '3by5fb': 3-by-5 spring lattice with fixed boundaries 
    ## 'sc20': spring chain

    for system_type in range(len(system_lst)):
        data_npy = np.load('./data/combined_data_' + str(system_lst[system_type]) + '_' + str(dataset_index) + '.npy') 

        train_data = torch.from_numpy(data_npy[:T, :, :])
        test_data = torch.from_numpy(data_npy[T:, :n_test_samples, :])

        for method in range(6): 
        ## 0: Leapfrog with MLP Hamiltonian
        ## 1: Leapfrog with MLP Time Derivative 
        ## 2: MLP for predicting p_{t+1}, q_{t+1} from p_t, q_t 
        ## 3: RNN
        ## 4: LSTM
        ## 5: Leapfrog with MLP for Separable Hamiltonian

            ## Training the model
            ## Case 1: p_0 is known
            # model = train(train_data, system_type=system_type, method=method, T=T, batch_size=batch_size, n_epochs=n_epochs, n_samples=n_samples, dt=dt, lr=lr, n_hidden=n_hidden, n_v=n_v, n_h=n_h, chain_length=chain_length, know_p_0=True)
            ## Case 2: p_0 is not known and has to be learned for each trajectory 
            model, p_0_var = train(train_data, system_type=system_type, method=method, T=T, batch_size=batch_size, n_epochs=n_epochs, n_samples=n_samples, dt=dt, lr=lr, n_hidden=n_hidden, n_v=n_v, n_h=n_h, chain_length=chain_length, know_p_0=False)

            ## Saving the model
            torch.save(model, './models/model_' + str(system_lst[system_type]) + '_' + str(method) + '_' + str(dataset_index) + '_' + str(run_index))
            
            ## Predicting the test trajectories
            ## Case 1: p_0 of the trajectories is known
            # traj_pred = predict(test_data, model=model, system_type=system_type, method=method, T_test=T_test, n_test_samples=n_test_samples, dt=dt, n_v = n_v, n_h = n_h, chain_length=chain_length, know_p_0 = True)
            ## Case 2: p_0 of the trajectories is not known
            traj_pred = predict(train_data[0, :n_test_samples, :], model=model, system_type=system_type, method=method, T_test=T+T_test, n_test_samples=n_test_samples, dt=dt, n_v = n_v, n_h = n_h, chain_length=chain_length, know_p_0 = False, p_0_pred = p_0_var[:n_test_samples, :].data)
            # traj_pred_cheat = predict(test_data, model=model, system_type=system_type, method=method, T_test=T_test, n_test_samples=n_test_samples, dt=dt, n_v = n_v, n_h = n_h, chain_length=chain_length, know_p_0 = True)

            print ('done making the prediction for' + str(system_lst[system_type]) + 'with method' + str(method))

            ## Saving the predicted trajectories
            np.save('./predictions/traj_pred_' + str(system_lst[system_type]) + '_' + str(method) + '_' + str(dataset_index) + '_' + str(run_index) + '.npy', traj_pred.data.numpy())

            print ('done saving the predicted trajectory')

if __name__ == "__main__":
    main()