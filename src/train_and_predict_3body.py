import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import time
import random
from torch.autograd import Variable, grad
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pickle
import sys
sys.path.append('../src')
# import utils.Logger
from utils import Logger, loss_plot, loss_plot_restricted, write_args
from os import path
from train_and_predict import numerically_integrate
from train_and_predict_chain import MLP1H, MLP2H, MLP3H, MLP2H_Separable_Hamilt, MLP_General_Hamilt, MLP1H_Separable_Hamilt, MLP3H_Separable_Hamilt, PlainRNN, myLSTM
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import minimize

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noiseless_dataset_index', nargs='?', type=str)
parser.add_argument('--train_noise_level', type=float, default=0.0)
parser.add_argument('--test_noise_level', type=float, default=0.0)
parser.add_argument('--run_index', type=str)
parser.add_argument('--T', type=int, default=10)
parser.add_argument('--dt', type=float, default=0.1)
parser.add_argument('--T_test', type=int, default=50)
parser.add_argument('--T_test_show', type=int, default=50)
parser.add_argument('--n_samples', type=int, default=1000)
parser.add_argument('--n_val_samples', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--lr_init', type=float, default=5e-3)
parser.add_argument('--leapfrog_train', type=str, default='true')
parser.add_argument('--leapfrog_test', type=str, default='true')
parser.add_argument('--scheduler_type', type=str, default='no_scheduler')
parser.add_argument('--shorten', type=int, default=0)
parser.add_argument('--T_short', type=int, default=0)
parser.add_argument('--scheduler_patience', type=int, default=10)
parser.add_argument('--scheduler_factor', type=float, default=0.5)
parser.add_argument('--T_init_seq', type=int, default=10)
# parser.add_argument('--n_iters_train_init', type=int, default=10)
parser.add_argument('--max_iters_init_train', type=int, default=10)
parser.add_argument('--max_iters_init_test', type=int, default=10)
parser.add_argument('--coarsening_factor_test', type=int, default=1)

args = parser.parse_args()

class PyTorchObjective(object):

    def __init__(self, true_data, model, dim, integrator, T, dt, device, method, batch_size=32, alpha=0.1, z_0_old=None):
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
        self.alpha = alpha
        self.noisy_z_0 = true_data[0, :].to('cpu').detach().numpy()
        if (z_0_old is None):
            self.z_0_old = true_data[0, :].to('cpu').detach().numpy()
        else:
            self.z_0_old = z_0_old

    # def is_new(self, x):
    #     # if this is the first thing we've seen
    #     if not hasattr(self, 'cached_x'):
    #         return True
    #     else:
    #         # compare x to cached_x to determine if we've been given a new input
    #         x, self.cached_x = np.array(x), np.array(self.cached_x)
    #         error = np.abs(x - self.cached_x)
    #         return error.max() > 1e-8

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
        # print ('grad', grad)
        return grad
        # self.loss = loss

    def fun_reg(self, z_0_npy):
        z_0 = torch.from_numpy(z_0_npy).type(torch.FloatTensor).unsqueeze(0)
        z_0.requires_grad_()
        z_0_device = z_0.to(self.device)
        p_0 = z_0_device[:, :self.dim]
        q_0 = z_0_device[:, self.dim:]
        trajectory_simulated = numerically_integrate(integrator=self.integrator, p_0=p_0, q_0=q_0, model=self.model, method=self.method, T=self.T, dt=self.dt, volatile=False, device=self.device)
        loss = self.mse(trajectory_simulated, self.true_data)
        # self.loss = loss
        loss_reg = loss + self.alpha * np.linalg.norm(z_0_npy - self.z_0_old)
        return loss_reg

    def jac_reg(self, z_0_npy):
        z_0 = torch.from_numpy(z_0_npy).type(torch.FloatTensor).unsqueeze(0)
        z_0.requires_grad_()
        z_0_device = z_0.to(self.device)
        p_0 = z_0_device[:, :self.dim]
        q_0 = z_0_device[:, self.dim:]
        trajectory_simulated = numerically_integrate(integrator=self.integrator, p_0=p_0, q_0=q_0, model=self.model, method=self.method, T=self.T, dt=self.dt, volatile=False, device=self.device)
        loss = self.mse(trajectory_simulated, self.true_data)
        loss.backward()
        grad = z_0.grad.type(torch.DoubleTensor).numpy()
        # print ('grad', grad)
        grad_reg = grad + 2 * self.alpha * (z_0_npy - self.z_0_old)
        return grad_reg
        # self.loss = loss


    def fun_batch(self, z_0_unrolled_npy):
        z_0 = torch.from_numpy(z_0_unrolled_npy).type(torch.FloatTensor).view(-1, self.dim * 2)
        z_0.requires_grad_()
        z_0_device = z_0.to(self.device)
        p_0 = z_0_device[:, :self.dim]
        q_0 = z_0_device[:, self.dim:]
        # if (self.integrator == 'NA'):
        #     trajectory_simulated = self.model(torch.cat([p_0, q_0], 1), self.T)
        # else:
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
        # if (self.integrator == 'NA'):
        #     trajectory_simulated = self.model(torch.cat([p_0, q_0], 1), self.T)
        # else:
        trajectory_simulated = numerically_integrate(integrator=self.integrator, p_0=p_0, q_0=q_0, model=self.model, method=self.method, T=self.T, dt=self.dt, volatile=False, device=self.device)
        loss = self.mse(trajectory_simulated, self.true_data)
        loss.backward()
        grad = z_0.grad.type(torch.DoubleTensor).view(z_0_unrolled_npy.shape[0]).numpy()
        # print ('grad', grad)
        return grad

    def fun_batch_reg(self, z_0_unrolled_npy):
        z_0_npy = z_0_unrolled_npy.reshape(-1, self.dim * 2)
        z_0 = torch.from_numpy(z_0_npy).type(torch.FloatTensor)#.view(-1, self.dim * 2)
        z_0.requires_grad_()
        z_0_device = z_0.to(self.device)
        p_0 = z_0_device[:, :self.dim]
        q_0 = z_0_device[:, self.dim:]
        trajectory_simulated = numerically_integrate(integrator=self.integrator, p_0=p_0, q_0=q_0, model=self.model, method=self.method, T=self.T, dt=self.dt, volatile=False, device=self.device)
        loss = self.mse(trajectory_simulated, self.true_data)
        # self.loss = loss
        loss_reg = loss + self.alpha * np.linalg.norm(z_0_npy - self.z_0_old)
        return loss_reg

    def jac_batch_reg(self, z_0_unrolled_npy):
        z_0 = torch.from_numpy(z_0_unrolled_npy).type(torch.FloatTensor).view(-1, self.dim * 2)
        z_0.requires_grad_()
        z_0_device = z_0.to(self.device)
        p_0 = z_0_device[:, :self.dim]
        q_0 = z_0_device[:, self.dim:]
        trajectory_simulated = numerically_integrate(integrator=self.integrator, p_0=p_0, q_0=q_0, model=self.model, method=self.method, T=self.T, dt=self.dt, volatile=False, device=self.device)
        loss = self.mse(trajectory_simulated, self.true_data)
        loss.backward()
        grad = z_0.grad.type(torch.DoubleTensor).view(z_0_unrolled_npy.shape[0]).numpy()
        grad_reg = grad + 2 * self.alpha * (z_0_unrolled_npy - self.z_0_old.reshape(z_0_unrolled_npy.shape[0]))
        # print ('grad', grad)
        return grad_reg



def train_initasparam(data, system_type, method, T, batch_size, n_epochs, n_samples, dt, lr, lr_init, n_layers, n_hidden, chain_length=20, know_p_0=True, integrator_train='leapfrog', integrator_test='leapfrog', regularization_constant=1, logger=None, device='cpu', test_data=None, n_val_samples=0, T_init_seq=10, n_test_samples=0, T_test=0, T_test_show=0, scheduler_type=None, scheduler_patience=200, scheduler_factor=0.5, pred_out_string=None, max_iters_init_train=10, max_iters_init_test=10, coarsening_factor_test=1, test_freq=1):

    dim = int(data.shape[2] / 2)

    if (method == 0):
        model = MLP_QuadKinetic_Hamilt(n_hidden, dim) #QuadHamilt()
    elif (method == 1) or (method == 21):
        if (n_layers == 1):
            model = MLP1H(2 * dim, n_hidden, 2 * dim)
        elif (n_layers == 2):
            model = MLP2H(2 * dim, n_hidden, 2 * dim)
        elif (n_layers == 3):
            model = MLP3H(2 * dim, n_hidden, 2 * dim)
    elif (method == 2):
        model = NaiveMLP(2 * dim, n_hidden, 2 * dim)
    elif (method == 3):
        model = PlainRNN(2 * dim, n_hidden, 2 * dim)
    elif (method == 4):
        model = myLSTM(2 * dim, 2 * dim)
    elif (method == 5) or (method == 25):
        if (n_layers == 1):
            model = MLP1H_Separable_Hamilt(n_hidden, dim)
        # model = MLP3H_Separable_Hamilt(n_hidden, dim)
        elif (n_layers == 2):
            model = MLP2H_Separable_Hamilt(n_hidden, dim)
        elif (n_layers == 3):
            model = MLP3H_Separable_Hamilt(n_hidden, dim)
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

    # z_0 = data[0, :, :].detach()
    # z_0 = torch.zeros(data[0, :, :].shape, requires_grad=False)
    # z_0_npy = np.zeros(data[0, :, :].shape, dtype=np.float32)
    z_0_npy = data[0, :, :].numpy()
    # z_0 = Parameter(data[0, :, :])
    # z_0.requires_grad_()

    # if (torch.cuda.is_available()):
    #     if (not ((method == 2) or (method == 3) or (method==4))):
    #         data = data.cuda()
    #         model = model.cuda()
    # if (not ((method == 2) or (method == 3) or (method==4))):
    if True:
        data = data.to(device)
        model = model.to(device)
        # z_0 = z_0.to(device)
        if (not test_data is None):
            test_data = test_data.to(device)

    # z_0.requires_grad_()

    # q_0 = data[0, :, dim:]
    # if know_p_0:
    #     p_0 = data[0, :, :dim]
    #     opt = torch.optim.Adam(model.parameters(), lr=lr)
    # else:
    #     p_0 = torch.rand(q_0.shape[0], dim)
    #     # p_0_var = Variable(p_0, requires_grad = True)
    #     # q_0_var = Variable(q_0, requires_grad = True)
    #     p_0_var = p_0.requires_grad_()
    #     q_0_var = q_0.requires_grad_()

        # params_lst = list(model.parameters())
        # params_lst.append(p_0_var)
        # opt = torch.optim.Adam(params_lst, lr=lr)

    # opt_init = torch.optim.Adam([z_0], lr=lr_init)

    params_lst = model.parameters()
    opt_model = torch.optim.Adam(params_lst, lr=lr)

    if (scheduler_type == 'plateau'):
        scheduler = ReduceLROnPlateau(opt_model, 'min', patience = scheduler_patience, verbose=True, factor=scheduler_factor)

    loss_record = []
    val_loss_record = []
    test_loss_record = []

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
            # z_0_permed = z_0[perm, :].clone()
            z_0_npy_permed = z_0_npy[perm, :]
            # if know_p_0:
            #     p_0_permed = p_0[perm, :]
            #     q_0_permed = q_0[perm, :]
            # else:
            #     p_0_var_permed = p_0_var[perm, :]
            #     q_0_var_permed = q_0_var[perm, :]

            loss_record_epoch = []

            for i in range(0, n_samples, batch_size):

                opt_model.zero_grad()
                # opt_init.zero_grad()


                if i + batch_size > n_samples:
                    break

                batch = data_permed[:max_t, i:(i+batch_size), :]#.to(device)
                z_0_batch_npy = z_0_npy_permed[i:(i+batch_size), :]#.clone()#.to(device)

                # print (z_0_batch.requires_grad)
                z_0_batch = torch.from_numpy(z_0_batch_npy)
                # z_0_batch.requires_grad_()

                # print (z_0_batch.requires_grad)

                # opt_init = torch.optim.Adam([z_0_batch], lr=lr_init)

                z_0_batch_device = z_0_batch.to(device)

                # print (z_0_batch_device.device)

                # device='cuda:0'

                # h = seq2init.initHidden().to(device)
                # for t in reversed(range(seq2init_length)):
                #     obs = batch[t, :, :]
                #     out, h = seq2init.forward(obs, h)

                p_0 = z_0_batch_device[:, :dim]
                q_0 = z_0_batch_device[:, dim:]

                # lf = True
                # if know_p_0:
                #     if (method == 0 or method == 5 or method == 7 or method == 9):
                #         if lf:
                #             trajectory_simulated = leapfrog(p_0, q_0, model, max_t, dt, volatile=False, device=device)
                #         else:
                #             trajectory_simulated = euler(p_0, q_0, model, max_t, dt, volatile=False, device=device)
                #     elif (method == 1 or method == 6 or method == 8 or method == 10):
                #         if lf:
                #             trajectory_simulated = leapfrog(p_0, q_0, model, max_t, dt, volatile=False, is_Hamilt=False, device=device)
                #         else:
                #             trajectory_simulated = euler(p_0, q_0, model, max_t, dt, volatile=False, is_Hamilt=False, device=device)
                #     else:
                #         trajectory_simulated = model(batch[0, :, :], max_t)

                # if (method == 3) or (method == 4):
                #     trajectory_simulated = model(torch.cat([p_0, q_0], 1), max_t)
                # else:
                trajectory_simulated = numerically_integrate(integrator=integrator_train, p_0=p_0, q_0=q_0, model=model, method=method, T=max_t, dt=dt, volatile=False, device=device, coarsening_factor=1)

                # else:
                #     if (method == 0 or method == 5 or method == 7 or method == 9):
                #         if lf:
                #             trajectory_simulated = leapfrog(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, device=device)
                #         else:
                #             trajectory_simulated = euler(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, device=device)
                #     elif (method == 1 or method == 6 or method == 8 or method == 10):
                #         if lf:
                #             trajectory_simulated = leapfrog(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, device=device)
                #         else:
                #             trajectory_simulated = euler(p_0_var_permed[i:(i+batch_size), :], q_0_var_permed[i:(i+batch_size), :], model, max_t, dt, volatile=False, is_Hamilt=False, device=device)
                #     else:
                #         trajectory_simulated = model(torch.cat([p_0_var_permed[i:(i+batch_size), :].data, q_0_var_permed[i:(i+batch_size), :].data], 1), max_t)

                ## If we know both the true p trajectory and the true q trajectory:
                # print (i, z_0_batch.requires_grad)
                error_total = mse(trajectory_simulated[:max_t, :, :], batch[:max_t, :, :])

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

                # print (z_0_batch.grad)

                opt_model.step()
                # print ('z_0_batch', z_0_batch.grad)
                # print ('z_0_batch_device', z_0_batch_device.grad)

                # if (epoch > 20):
                #     opt_init.step()
                    # z_0_npy[perm, :] = z_0_npy_permed

                # print ('z_0_batch',z_0_batch)
                # print ('z_0_batch_npy', z_0_batch_npy)
                # print ('z_0_permed', z_0_permed)

                # print ('z_0', z_0)



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

            alpha = T * 0
            if (max_iters_init_train > 0) and (epoch > 100):# and (epoch % 5 == 0):
                print ('muha')
                # for j in range(n_samples):
                #     objective = PyTorchObjective(data[:, j, :], model, dim, integrator=integrator_train, T=T, dt=dt, device=device, method=method, alpha=alpha)
                #     print (z_0_npy.shape)
                #     result = minimize(objective.fun_reg, z_0_npy[j, :], method='L-BFGS-B', jac=objective.jac_reg,
                #          options={'gtol': 1e-6, 'disp': False,
                #         'maxiter':10})
                #     print ('old', z_0_npy[j, :], 'new', result.x)
                #     print ('diff', z_0_npy[j, :] - result.x)
                #     z_0_npy[j, :] = result.x
                objective = PyTorchObjective(data[:T, :, :].detach(), model, dim, integrator=integrator_train, T=T, dt=dt, device=device, method=method, alpha=alpha, z_0_old=z_0_npy)
                z_0_unrolled_npy = z_0_npy.reshape(n_samples * dim * 2)
                result = minimize(objective.fun_batch, z_0_unrolled_npy, method='L-BFGS-B', jac=objective.jac_batch,
                         options={'gtol': 1e-6, 'disp': False,
                        'maxiter':max_iters_init_train})
                # print (max_iters_init_train)
                z_0_npy_new = result.x.reshape(n_samples, dim * 2)
                # print ('old', z_0_npy, 'new', z_0_npy_new
                print ('diff', np.mean(np.abs(z_0_npy - z_0_npy_new)))
                z_0_npy = z_0_npy_new.astype(np.float32).copy()

            if (epoch % 1 == 0):
                # ## Get validation error for scheduler
                T_val = T
                val_indices = torch.randperm(n_samples)[:n_val_samples].tolist()
                # print ('val indices', val_indices.shape)
                # val_indices = val_indices[:n_val_samples].tolist()
                # print ('val indices', val_indices)
                val_data = data[:, val_indices, :]
                z_0_val_npy = z_0_npy[val_indices, :]
                z_0_val = torch.from_numpy(z_0_val_npy)
                # z_0_val.requires_grad_()
                z_0_val_device = z_0_val.to(device)
                # print ('val data', val_data.size())
                # val_data_init_seq = val_data[:seq2init_length, :, :]
                # pred_val = predict_initasparam(z_0=z_0_val, test_data_init_seq=None, model=model, method=method, T_test=T_val, n_test_samples=n_val_samples, dt=dt, integrator=integrator_train, device=device, n_epochs_train_init=20)
                # if (method == 3) or (method == 4):
                #     pred_val = model(z_0_val_device, T_val)
                # else:
                pred_val = numerically_integrate(integrator=integrator_train, p_0=z_0_val_device[:, :dim], q_0=z_0_val_device[:, dim:], model=model, method=method, T=T_val, dt=dt, volatile=True, device=device)
                # val_error = mse(pred_val[:T_val, :, dim:], val_data[:T_val, :n_val_samples, dim:])
                val_error = mse(pred_val[:T_val, :, :], val_data[:T_val, :n_val_samples, :])
                # print(epoch, i, max_t, 'val', val_error.item())
                val_loss_record.append(val_error.item())
                # else:

                if not (scheduler_type == 'no_scheduler'):
                	scheduler.step(val_error.cpu().data.numpy())

                # ## Get test error
            if (epoch % test_freq == 0):
                test_data_init_seq = test_data[:T_init_seq, :, :]
                pred_test = predict_initasparam(test_data_init_seq=test_data_init_seq, model=model, method=method, T_test=T_test, n_test_samples=n_test_samples, dt=dt, integrator=integrator_test, device=device, max_iters_init=max_iters_init_test, lr_init=lr_init, coarsening_factor=coarsening_factor_test)
                # val_error = mse(pred_val[:T_val, :, dim:], val_data[:T_val, :n_val_samples, dim:])
                test_error = mse(pred_test[:T_test_show, :, :], test_data[:T_test_show, :n_test_samples, :])
                # print(epoch, i, max_t, 'val', val_error.item(), 'test', test_error.item())
                test_loss_record.append(test_error.item())

                ## Save predicted trajectory (on test dataset)
                if (not pred_out_string is None) and (epoch % 50 == 0):
                    np.save(pred_out_string + '_ep' + str(epoch) + '.npy', pred_test.detach().to('cpu'))


                if (logger is None):
                    # print(epoch, i, max_t, error_total.data[0])
                    print(epoch, i, max_t, 'trloss', error_total.item(), 'valoss', val_error.item(), 'teloss', test_error.item())
                    # "epoch %d train: p q seq loss %.6f image loss %.6f init p q prediction loss %.6f train loss %.6f" %
                    # ((t,) + tuple(np.array(local_loss_record).mean(axis=0)))
                    # print ("epoch %d i %d max_t %d loss %.6f" % (epoch, i, max_t, error_total.item()))
                else:
                    # logger.log(epoch, i, max_t, error_total.item())
                    logger.log("method %d epoch %d max_t %d trloss %.6f valoss %.6f teloss %.6f" % (method, epoch, max_t, avg_loss_epoch, val_error.item(), test_error.item()))


    if know_p_0:
        return model, loss_record, test_loss_record
    else:
        return model, p_0_var, loss_record#, loss_record_val


def predict_initasparam(test_data_init_seq, model, method, T_test, n_test_samples, dt, chain_length=20, coarsen=False, integrator='leapfrog', device='cpu', n_iters_train_init=20, lr_init=5e-3, max_iters_init=10, coarsening_factor=1):
    model = model.to(device)
    test_data_init_seq = test_data_init_seq.to(device)

    # if (z_0 is None):
    dim = int(test_data_init_seq.shape[2] / 2)
    T_init_seq = test_data_init_seq.size(0)

    z_0_npy = test_data_init_seq[0, :n_test_samples, :].to('cpu').numpy()

    # z_0 = z_0.to(device)

    # print (device)
    # print (z_0.device)

    # z_0.requires_grad_()

    # print (z_0.device)

    # opt_init = torch.optim.Adam([z_0], lr=lr_init)


    # mse = nn.MSELoss()

    # for epoch in range(n_epochs_train_init):
    #     trajectory_simulated = numerically_integrate(integrator=integrator, p_0=z_0[:, :dim], q_0=z_0[:, dim:], model=model, method=method, T=T_init_seq, dt=dt, volatile=False, device=device)
    #     error = mse(trajectory_simulated, test_data_init_seq.requires_grad_())
    #     error.backward()
    #     opt_init.step()
    # for i in range(n_epochs_train_init):

    if (max_iters_init > 0):
        alpha=T_test * 0
        # if (method == 3) or (method == 4):
        #     objective = PyTorchObjective(test_data_init_seq.detach(), model, dim, integrator='NA', T=T_init_seq, dt=dt, device=device, method=method, alpha=alpha)
        # else:
        objective = PyTorchObjective(test_data_init_seq.detach(), model, dim, integrator=integrator, T=T_init_seq, dt=dt, device=device, method=method, alpha=alpha)
        z_0_unrolled_npy = z_0_npy.reshape(n_test_samples * dim * 2)
        # result = minimize(objective.fun_batch, z_0_unrolled_npy, method='L-BFGS-B', jac=objective.jac_batch,
        #          options={'gtol': 1e-6, 'disp': False,
        #         'maxiter':n_iters_train_init})
        result = minimize(objective.fun_batch, z_0_unrolled_npy, method='L-BFGS-B', jac=objective.jac_batch,
                 options={'gtol': 1e-6, 'disp': False,
                'maxiter':max_iters_init})
        z_0_npy_new = result.x.reshape(n_test_samples, dim * 2)
        # print ('old', z_0_npy, 'new', z_0_npy_new
        print ('diff', np.mean(np.abs(z_0_npy - z_0_npy_new)))
        z_0_npy = z_0_npy_new.astype(np.float32).copy()

    z_0 = torch.from_numpy(z_0_npy).to(device)

    # else:
    #     dim = int(z_0.shape[2] / 2)
    #     z_0 = z_0[:n_test_samples, :]
    #     z_0.to(device)

    # # if (torch.cuda.is_available()):
    # #     if (not ((method == 2) or (method == 3) or (method==4))):
    # #         test_data_init_seq = test_data_init_seq.cuda()

    # # seq2init.nbatch = test_data_init_seq.size(1)
    # # h = seq2init.initHidden().to(device)
    # # for t in reversed(range(seq2init_length)):
    # #     obs = test_data_init_seq[t, :, :]
    # #     out, h = seq2init.forward(obs, h)

    # p_0 = z_0[:, :dim]
    # q_0 = z_0[:, dim:]

    # # q_0 = init_state[:, dim:]
    # # if know_p_0:
    # #     # p_0 = test_data[0, :, :dim]
    # #     p_0 = init_state[:, :dim]
    # # else:
    # #     p_0 = p_0_pred

    # # if (method == 0 or method == 5 or method == 7 or method == 9):
    # #     if coarsen:
    # #         coarsening_factor = 10
    # #         if lf:
    # #             fine_trajectory = leapfrog(p_0, q_0, model, T_test * coarsening_factor, dt / coarsening_factor, volatile=False, device=device)
    # #         else:
    # #             fine_trajectory = euler(p_0, q_0, model, T_test * coarsening_factor, dt / coarsening_factor, volatile=False, device=device)
    # #         trajectory_predicted = fine_trajectory[np.arange(T_test) * coarsening_factor, :, :]
    # #     else:
    # #         if lf:
    # #             trajectory_predicted = leapfrog(p_0, q_0, model, T_test, dt, volatile=False, device=device)
    # #         else:
    # #             trajectory_predicted = euler(p_0, q_0, model, T_test, dt, volatile=False, device=device)
    # # elif (method == 1 or method == 6 or method == 8 or method == 10):
    # #     if lf:
    # #         trajectory_predicted = leapfrog(p_0, q_0, model, T_test, dt, volatile=False, is_Hamilt=False, device=device)
    # #     else:
    # #         trajectory_predicted = euler(p_0, q_0, model, T_test, dt, volatile=False, is_Hamilt=False, device=device)
    # # else:
    # #     # trajectory_predicted = model(test_data[0, :, :], T_test * 2)
    # #     trajectory_predicted = model(torch.cat([p_0, q_0], 1), T_test)

    # if (method == 3) or (method == 4):
    #     trajectory_predicted = model(z_0, T_test)
    # else:
    trajectory_predicted = numerically_integrate(integrator=integrator, p_0=z_0[:, :dim], q_0=z_0[:, dim:], model=model, method=method, T=T_test, dt=dt, volatile=True, device=device, coarsening_factor=coarsening_factor)

    return trajectory_predicted


def main():

    device = 'cuda:0'

    # dataset_index = 'H'
    # noiseless_dataset_index = 'B'
    noiseless_dataset_index = args.noiseless_dataset_index
    # train_noise_level = 0.1
    train_noise_level = args.train_noise_level
    test_noise_level = args.test_noise_level
    if (train_noise_level > 0):
    	dataset_index = noiseless_dataset_index + 'n' + str(train_noise_level)
    else:
    	dataset_index = noiseless_dataset_index# + '_cf100'
    if (test_noise_level > 0):
        test_dataset_index = noiseless_dataset_index + 'n' + str(test_noise_level)
    else:
        test_dataset_index = noiseless_dataset_index #+ '_cf100'
    # run_index = 'r9'
    run_index = args.run_index

    ## Parameters of the learning process
    # batch_size = 32  ## batch size
    batch_size = args.batch_size
    # n_epochs = 200  ## number of epochs in training
    # n_epochs = 200
    # n_epochs = 100
    # n_epochs = 40
    n_epochs = args.n_epochs
    # T = 100  ## time length of training trajectories
    # T = 200
    # T = 4000
    # T = 20
    T = args.T

    # dt = 0.001  ## time step of numerical discretization
    # dt = 0.1
    dt = args.dt
    # lr = 5 * 1e-3  ## learning rate
    # lr = 5 * 1e-4
    # lr = 1e-1
    lr = args.lr
    lr_init = args.lr_init
    n_layers = args.n_layers
    # n_hidden=256  ## number of hidden units in the MLP's or RNN's
    # n_hidden = 16
    n_hidden = args.n_hidden
    # n_samples = 2000  ## number of training samples
    # n_samples = 512
    # n_samples = 448
    # n_samples = 384
    # n_samples = 944
    # n_samples = 32
    # n_samples = 1024
    # n_samples = 1024
    # n_samples = 1000
    n_samples = args.n_samples
    # n_test_samples = 30  ## number of testing samples
    n_test_samples = 32
    n_val_samples = args.n_val_samples
    # T_test = 1000  ## time length of testing samples
    # T_test = 2000
    # T_test = 1000
    # T_test = 50
    T_test = args.T_test
    T_test_show = args.T_test_show

    know_p_0 = True

    # leapfrog_train = args.leapfrog_train
    # leapfrog_test = args.leapfrog_test

    if (args.leapfrog_train == 'true'):
        integrator_train = 'leapfrog'
    else:
        integrator_train = 'euler'
    if (args.leapfrog_test == 'true'):
        integrator_test = 'leapfrog'
    else:
        integrator_test = 'euler'

    shorten = args.shorten

    # T_short = 4
    # T_short = 2
    T_short = args.T_short

    T_total = T + T_test

    regularization_constant = 1

    # seq2init_length = args.seq2init_length
    # seq2init_n_hidden = args.seq2init_n_hidden
    T_init_seq = args.T_init_seq

    # n_iters_train_init = args.n_iters_train_init
    max_iters_init_train = args.max_iters_init_train
    max_iters_init_test = args.max_iters_init_test

    scheduler_type = args.scheduler_type
    scheduler_patience = args.scheduler_patience
    scheduler_factor = args.scheduler_factor

    coarsening_factor_test = args.coarsening_factor_test

    test_freq = 10


    #### Training the models and making the predictions

    data_dir = './data/3body'
    model_dir = '../models/3body_' + str(dataset_index) + str(run_index)
    pred_dir = '../predictions/3body_' + str(dataset_index) + str(run_index)
    log_dir_together = '../logs/3body_' + str(dataset_index) + str(run_index)

    if (not os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    ## List of systems to be learned
    system_lst = ['3body']



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

    write_args(args,os.path.join(log_dir_together))
    print(vars(args))

    for system_type in range(len(system_lst)):

        train_data_npy = np.load(data_dir + '/train_data_3body_' + dataset_index + '.npy')
        test_data_npy = np.load(data_dir + '/test_data_3body_' + test_dataset_index  + '.npy')

        train_data = torch.as_tensor(train_data_npy[:, :n_samples, :], dtype=torch.float32)
        test_data = torch.as_tensor(test_data_npy[:, :n_test_samples, :], dtype=torch.float32)


        if (dt == 0.1):
            print ('coarsening to  dt=0.1')
            train_data = train_data[np.arange(T) * 10, :, :]
            test_data = test_data[np.arange(T) * 10, :, :]

        elif (dt == 1):
            print ('coarsening to  dt=0.1')
            train_data = train_data[np.arange(T) * 100, :, :]
            test_data = test_data[np.arange(T) * 100, :, :]

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
            train_data_shortened = torch.zeros(max(T_short, T_init_seq), n_samples * (T - max(T_short, T_init_seq) + 1), train_data_npy.shape[2])
            # for i in range(int(T / T_short)):
            for i in range(T - max(T_short, T_init_seq) + 1):
                # train_data_shortened[:, i * n_samples : (i+1) * n_samples, :] = train_data[i * (T_short) : (i+1) * T_short, :n_samples, :]
                train_data_shortened[:, i * n_samples : (i+1) * n_samples, :] = train_data[i : i + max(T_short, T_init_seq), :n_samples, :]

            train_data = train_data_shortened
            T = T_short
            n_samples = train_data_shortened.shape[1]

            print ('new num samples', n_samples)


        # init_test_data = test_data[0, :, :]

        # for method in [8, 10, 5, 7, 1, 4, 3]:
        for method in [5, 1]:
        ## 0: Leapfrog with MLP Hamiltonian
        ## 1: Leapfrog with MLP Time Derivative
        ## 2: MLP for predicting p_{t+1}, q_{t+1} from p_t, q_t
        ## 3: RNN
        ## 4: LSTM
        ## 5: Leapfrog with MLP for Separable Hamiltonian

            log_dir = '../logs/3body_' + str(dataset_index) + str(run_index) + '/m' + str(method)
            pred_out_string = pred_dir + '/traj_pred_' + str(system_lst[system_type]) + '_' + str(method) + '_' + str(test_dataset_index) + '_' + str(run_index)

            if (not os.path.isdir(log_dir)):
                os.mkdir(log_dir)

            logger0 = Logger(os.path.join(log_dir, 'trloss.log'), print_out=True)
            logger1 = Logger(os.path.join(log_dir, 'teloss.log'), print_out=True)

            start = time.time()

            ## Training the model
            if know_p_0:
                ## Case 1: p_0 is known
                # model = train(train_data, system_type=system_type, method=method, T=T, batch_size=batch_size, n_epochs=n_epochs, n_samples=n_samples, dt=dt, lr=lr, n_hidden=n_hidden, know_p_0=True, lf=leapfrog)
                model, loss_record, val_loss_record = train_initasparam(train_data, system_type=system_type, method=method, T=T, batch_size=batch_size, n_epochs=n_epochs, n_samples=n_samples, n_val_samples=args.n_val_samples, dt=dt, lr=lr, lr_init=lr_init, n_layers=n_layers, n_hidden=n_hidden, integrator_train=integrator_train, integrator_test=integrator_test, regularization_constant=regularization_constant, logger=logger0, device=device, test_data=test_data, n_test_samples=n_test_samples, T_test=T_test, T_test_show=T_test_show, scheduler_type=scheduler_type, scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor, pred_out_string=pred_out_string, max_iters_init_train=max_iters_init_train, max_iters_init_test=max_iters_init_test, T_init_seq=T_init_seq, coarsening_factor_test=coarsening_factor_test, test_freq=test_freq)

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
            loss_plot(os.path.join(log_dir, 'loss.pkl'), log_dir, name=['','','loss p','loss q'], teplotfreq = test_freq)
            loss_plot_restricted(os.path.join(log_dir, 'loss.pkl'), log_dir, name=['','','loss p','loss q'], teplotfreq = test_freq)

            train_time = time.time() - start
            print ('training with method ' + str(method) + ' costs time ', train_time)

            ## Predicting the test trajectories
            if know_p_0:
                ## Case 1: p_0 of the trajectories is known
                # traj_pred = predict_initasparam(test_data[:seq2init_length, :, :], model=model, seq2init=seq2init, system_type=system_type, method=method, T_test=T_total, n_test_samples=n_test_samples, dt=dt, know_p_0 = True, lf=leapfrog_test, device='cuda:0', seq2init_length=seq2init_length)
                traj_pred = predict_initasparam(test_data_init_seq=test_data[:T_init_seq, :, :], model=model, method=method, T_test=T_test, n_test_samples=n_test_samples, dt=dt, integrator=integrator_test, device=device, lr_init=lr_init, max_iters_init=max_iters_init_test, coarsening_factor=coarsening_factor_test)
            else:
                ## Case 2: p_0 of the trajectories is not known
                traj_pred = predict(init_test_data, model=model,system_type=system_type, method=method, T_test=T_total, n_test_samples=n_test_samples, dt=dt, know_p_0 = False, p_0_pred = p_0_var[:n_test_samples, :].data, lf=leapfrog_test, device='cuda:0')
                # traj_pred_cheat = predict(test_data, model=model, system_type=system_type, method=method, T_test=T_test, n_test_samples=n_test_samples, dt=dt, n_v = n_v, n_h = n_h, chain_length=chain_length, know_p_0 = True)

            pred_time = time.time() - start
            print ('making the predictions with method ' + str(method) + ' costs time ', pred_time)
            # print ('done making the prediction for' + str(system_lst[system_type]) + 'with method' + str(method))

            if (torch.cuda.is_available()):
                np.save(pred_out_string + '.npy', traj_pred.cpu().data.numpy())
            else:
                ## Saving the predicted trajectories
                np.save(pred_out_string + '.npy', traj_pred.data.numpy())

            print ('done saving the predicted trajectory')


if __name__ == "__main__":
    main()
