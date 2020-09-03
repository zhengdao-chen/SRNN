# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Earlier versions of this file were written by Zhengdao Chen, used with permission.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import time
import random
from tqdm import tqdm
import pickle
import sys
sys.path.append('../src')
from utils import Logger, loss_plot, loss_plot_restricted, write_args
from os import path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import minimize

from model import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--system_type', nargs='?', type=str, help="'chain' or '3body'")
parser.add_argument('--noiseless_dataset_index', nargs='?', type=str, help="dataset_index from generate.py")
parser.add_argument('--train_noise_level', type=float, default=0.0, help="noise level of training data")
parser.add_argument('--test_noise_level', type=float, default=0.0, help="noise level of testing data")
parser.add_argument('--run_index', type=str, help="just to index each run")
parser.add_argument('--model_type', type=str, default='HNET', help="one of 'HNET', 'ONET', 'RNN' and 'LSTM'")
parser.add_argument('--T', type=int, default=10, help='number of time-steps of training trajectories')
parser.add_argument('--dt', type=float, default=0.1, help='size of time-step')
parser.add_argument('--T_test', type=int, default=50, help='number of time-steps of testing trajectories')
parser.add_argument('--n_samples', type=int, default=1000, help='number of training samples')
parser.add_argument('--n_val_samples', type=int, default=128, help='number of validation samples')
parser.add_argument('--n_test_samples', type=int, default=32, help='number of testing samples')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--n_layers', type=int, default=2, help='number of hidden layers of the MLP (1, 2 and 3 are allowed)')
parser.add_argument('--n_hidden', type=int, default=256, help='number of hidden units in the MLP/RNN/LSTM')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--leapfrog_train', type=str, default='true', help="whether to leapfrog or Euler in training ('true' and 'false' are allowed)")
parser.add_argument('--leapfrog_test', type=str, default='true', help="whether to leapfrog or Euler in testing ('true' and 'false' are allowed)")
parser.add_argument('--scheduler_type', type=str, default='no_scheduler', help="type of learning rate scheduler from PyTorch")
parser.add_argument('--shorten', type=int, default=0, help='whether to augment the number of training trajectories by shortening their lengths (1 and 2 correspond to two different ways of doing this; 0 for not doing so)')
parser.add_argument('--T_short', type=int, default=0, help='the new lengths of the training trajectories, if args.shorten is set to be 1 or 2')
parser.add_argument('--scheduler_patience', type=int, default=10, help='patience of learning rate scheduler')
parser.add_argument('--scheduler_factor', type=float, default=0.5, help='decay factor of learning rate scheduler')
parser.add_argument('--T_init_seq', type=int, default=10, help='length of the training & testing trajectories to which initial state optimization (ISO) is applied')
parser.add_argument('--max_iters_init_train', type=int, default=10, help='max number of L-BFGS-B steps for initial state optimization (ISO) in training; 0 if no ISO is wanted')
parser.add_argument('--max_iters_init_test', type=int, default=10, help='max number of L-BFGS-B steps for initial state optimization (ISO) in testing; 0 if no ISO is wanted')
parser.add_argument('--coarsening_factor_test', type=int, default=1, help='coarsening factor when predicting the trajectories for the testing data')
parser.add_argument('--test_freq', type=int, default=1, help='how many epochs to wait between consecutive testings')

args = parser.parse_args()




def train(data, method, T, batch_size, n_epochs, n_samples, dt, lr, n_layers, n_hidden, \
    chain_length=20, integrator_train='leapfrog', integrator_test='leapfrog', logger=None, device='cpu', \
    test_data=None, n_val_samples=0, T_init_seq=10, n_test_samples=0, T_test=0, scheduler_type=None, \
    scheduler_patience=200, scheduler_factor=0.5, pred_out_string=None, max_iters_init_train=10, \
    max_iters_init_test=10, coarsening_factor_test=1, test_freq=1):

    dim = int(data.shape[2] / 2)

    if (method == 1):
        if (n_layers == 1):
            model = MLP1H(2 * dim, n_hidden, 2 * dim)
        elif (n_layers == 2):
            model = MLP2H(2 * dim, n_hidden, 2 * dim)
        elif (n_layers == 3):
            model = MLP3H(2 * dim, n_hidden, 2 * dim)
    elif (method == 3):
        model = RNN(2 * dim, n_hidden, 2 * dim)
    elif (method == 4):
        model = LSTM(2 * dim, 2 * dim)
    elif (method == 5):
        if (n_layers == 1):
            model = MLP1H_Separable_Hamilt(n_hidden, dim)
        elif (n_layers == 2):
            model = MLP2H_Separable_Hamilt(n_hidden, dim)
        elif (n_layers == 3):
            model = MLP3H_Separable_Hamilt(n_hidden, dim)

    mse = nn.MSELoss()

    z_0_npy = data[0, :, :].numpy()

    data = data.to(device)
    model = model.to(device)
    if (not test_data is None):
        test_data = test_data.to(device)

    params_lst = model.parameters()
    opt_model = torch.optim.Adam(params_lst, lr=lr)

    if (scheduler_type == 'plateau'):
        scheduler = ReduceLROnPlateau(opt_model, 'min', patience = scheduler_patience, verbose=True, factor=scheduler_factor)

    loss_record = []
    val_loss_record = []
    test_loss_record = []

    for epoch in tqdm(range(n_epochs)):

        max_t = T

        if (max_t > 1):
            perm = torch.randperm(n_samples).numpy().tolist()
            data_permed = data[:, perm, :]
            z_0_npy_permed = z_0_npy[perm, :]

            loss_record_epoch = []

            for i in range(0, n_samples, batch_size):

                opt_model.zero_grad()

                if i + batch_size > n_samples:
                    break

                batch = data_permed[:max_t, i:(i+batch_size), :]
                z_0_batch_npy = z_0_npy_permed[i:(i+batch_size), :]

                z_0_batch = torch.from_numpy(z_0_batch_npy)

                z_0_batch_device = z_0_batch.to(device)

                p_0 = z_0_batch_device[:, :dim]
                q_0 = z_0_batch_device[:, dim:]

                trajectory_simulated = numerically_integrate(integrator=integrator_train, p_0=p_0, q_0=q_0, model=model, \
                    method=method, T=max_t, dt=dt, volatile=False, device=device, coarsening_factor=1)

                error_total = mse(trajectory_simulated[:max_t, :, :], batch[:max_t, :, :])

                error_total.backward()

                # For RNN, clip the gradients
                if(method == 3):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

                opt_model.step()

                loss_record_epoch.append(error_total.item())

            try:
                avg_loss_epoch = sum(loss_record_epoch) / len(loss_record_epoch)
            except ZeroDivisionError as e:
                print ("Error: ", e, "(batch size larger than number of training samples)")

            loss_record.append(avg_loss_epoch)


            ## initial state optimization
            if (max_iters_init_train > 0) and (epoch > 100):
                objective = PyTorchObjective(data[:T, :, :].detach(), model, dim, integrator=integrator_train, T=T, dt=dt, \
                    device=device, method=method, z_0_old=z_0_npy)
                z_0_unrolled_npy = z_0_npy.reshape(n_samples * dim * 2)
                result = minimize(objective.fun_batch, z_0_unrolled_npy, method='L-BFGS-B', jac=objective.jac_batch,
                         options={'gtol': 1e-6, 'disp': False,
                        'maxiter':max_iters_init_train})
                z_0_npy_new = result.x.reshape(n_samples, dim * 2)
                print ('diff', np.mean(np.abs(z_0_npy - z_0_npy_new)))
                z_0_npy = z_0_npy_new.astype(np.float32).copy()

            ## compute validation error for the scheduler to adjust lr
            if (epoch % 1 == 0):
                T_val = T
                val_indices = torch.randperm(n_samples)[:n_val_samples].tolist()
                val_data = data[:, val_indices, :]
                z_0_val_npy = z_0_npy[val_indices, :]
                z_0_val = torch.from_numpy(z_0_val_npy)
                z_0_val_device = z_0_val.to(device)
                pred_val = numerically_integrate(integrator=integrator_train, p_0=z_0_val_device[:, :dim], q_0=z_0_val_device[:, dim:], \
                    model=model, method=method, T=T_val, dt=dt, volatile=True, device=device)
                val_error = mse(pred_val[:T_val, :, :], val_data[:T_val, :n_val_samples, :])
                val_loss_record.append(val_error.item())

                if not (scheduler_type == 'no_scheduler'):
                	scheduler.step(val_error.cpu().data.numpy())

            ## compute test error
            if (epoch % test_freq == 0):
                test_data_init_seq = test_data[:T_init_seq, :, :]
                pred_test = predict(test_data_init_seq=test_data_init_seq, model=model, method=method, T_test=T_test, \
                    n_test_samples=n_test_samples, dt=dt, integrator=integrator_test, device=device, max_iters_init=max_iters_init_test, \
                    coarsening_factor=coarsening_factor_test)
                test_error = mse(pred_test[:T_test, :, :], test_data[:T_test, :n_test_samples, :])
                test_loss_record.append(test_error.item())

                ## Save predicted trajectory (on test dataset)
                if (not pred_out_string is None) and (epoch % 25 == 0):
                    np.save(pred_out_string + '_ep' + str(epoch) + '.npy', pred_test.detach().to('cpu'))

                if (logger is None):
                    print(epoch, i, max_t, 'trloss', error_total.item(), 'valoss', val_error.item(), 'teloss', test_error.item())
                else:
                    logger.log("method %d epoch %d max_t %d trloss %.6f valoss %.6f teloss %.6f" % (method, epoch, max_t, avg_loss_epoch, val_error.item(), test_error.item()))


    return model, loss_record, test_loss_record


def predict(test_data_init_seq, model, method, T_test, n_test_samples, dt, chain_length=20, integrator='leapfrog', \
    device='cpu', n_iters_train_init=20, max_iters_init=10, coarsening_factor=1):

    model = model.to(device)
    test_data_init_seq = test_data_init_seq.to(device)

    dim = int(test_data_init_seq.shape[2] / 2)
    T_init_seq = test_data_init_seq.size(0)

    z_0_npy = test_data_init_seq[0, :n_test_samples, :].to('cpu').numpy()

    if (max_iters_init > 0):
        objective = PyTorchObjective(test_data_init_seq.detach(), model, dim, integrator=integrator, T=T_init_seq, dt=dt, device=device, method=method)
        z_0_unrolled_npy = z_0_npy.reshape(n_test_samples * dim * 2)
        result = minimize(objective.fun_batch, z_0_unrolled_npy, method='L-BFGS-B', jac=objective.jac_batch,
                 options={'gtol': 1e-6, 'disp': False,
                'maxiter':max_iters_init})
        z_0_npy_new = result.x.reshape(n_test_samples, dim * 2)
        print ('diff', np.mean(np.abs(z_0_npy - z_0_npy_new)))
        z_0_npy = z_0_npy_new.astype(np.float32).copy()

    z_0 = torch.from_numpy(z_0_npy).to(device)

    trajectory_predicted = numerically_integrate(integrator=integrator, p_0=z_0[:, :dim], q_0=z_0[:, dim:], model=model, \
        method=method, T=T_test, dt=dt, volatile=True, device=device, coarsening_factor=coarsening_factor)

    return trajectory_predicted


def main():

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    system_type = args.system_type
    noiseless_dataset_index = args.noiseless_dataset_index
    train_noise_level = args.train_noise_level
    test_noise_level = args.test_noise_level
    run_index = args.run_index
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    T = args.T
    dt = args.dt
    lr = args.lr
    n_layers = args.n_layers
    n_hidden = args.n_hidden
    n_samples = args.n_samples
    n_test_samples = args.n_test_samples
    n_val_samples = args.n_val_samples
    T_test = args.T_test
    shorten = args.shorten
    T_short = args.T_short
    T_total = T + T_test
    T_init_seq = args.T_init_seq
    max_iters_init_train = args.max_iters_init_train
    max_iters_init_test = args.max_iters_init_test
    scheduler_type = args.scheduler_type
    scheduler_patience = args.scheduler_patience
    scheduler_factor = args.scheduler_factor
    coarsening_factor_test = args.coarsening_factor_test
    test_freq = args.test_freq

    if (train_noise_level > 0):
        dataset_index = noiseless_dataset_index + '_n' + str(train_noise_level)
    else:
        dataset_index = noiseless_dataset_index
    if (test_noise_level > 0):
        test_dataset_index = noiseless_dataset_index + '_n' + str(test_noise_level)
    else:
        test_dataset_index = noiseless_dataset_index

    data_dir = './data/' + system_type
    model_dir = './models/' + system_type + '_' + str(dataset_index) + str(run_index)
    pred_dir = './predictions/' + system_type + '_' + str(dataset_index) + str(run_index)
    log_dir_together = './logs/' + system_type + '_' + str(dataset_index) + str(run_index)

    if (not os.path.isdir(data_dir)):
        os.mkdir(data_dir)
    if (not os.path.isdir(model_dir)):
        os.mkdir(model_dir)
    if (not os.path.isdir(pred_dir)):
        os.mkdir(pred_dir)
    if (not os.path.isdir(log_dir_together)):
        os.mkdir(log_dir_together)

    print ('dataset', dataset_index)
    print ('run', run_index)

    write_args(args, os.path.join(log_dir_together))
    print(vars(args))

    train_data_npy = np.load(data_dir + '/train_data_' + system_type + '_' + dataset_index + '.npy')
    test_data_npy = np.load(data_dir + '/test_data_' + system_type + '_' + test_dataset_index  + '.npy')

    train_data = torch.from_numpy(train_data_npy[:, :n_samples, :])
    test_data = torch.from_numpy(test_data_npy[:, :n_test_samples, :])

    if (system_type == '3body'):
        if (dt == 0.1):
            print ('coarsening to  dt=0.1')
            train_data = train_data[np.arange(T) * 10, :, :]
            test_data = test_data[np.arange(T_test) * 10, :, :]
        elif (dt == 1):
            print ('coarsening to  dt=0.1')
            train_data = train_data[np.arange(T) * 100, :, :]
            test_data = test_data[np.arange(T_test) * 100, :, :]
    else:
        train_data = train_data[:T, :, :]
        test_data = test_data[:T_test, :, :]
    
    ## augmenting the number of training trajectories while shortening their lengths
    if (shorten == 1):
        train_data_shortened = torch.zeros(T_short, n_samples * int(T / T_short), train_data_npy.shape[2])
        for i in range(int(T / T_short)):
            train_data_shortened[:, i * n_samples : (i+1) * n_samples, :] = train_data[i * (T_short) : (i+1) * T_short, :n_samples, :]
        train_data = train_data_shortened
        T = T_short
        n_samples = train_data_shortened.shape[1]
    elif(shorten == 2):
        train_data_shortened = torch.zeros(max(T_short, T_init_seq), n_samples * (T - max(T_short, T_init_seq) + 1), train_data_npy.shape[2])
        for i in range(T - max(T_short, T_init_seq) + 1):
            train_data_shortened[:, i * n_samples : (i+1) * n_samples, :] = train_data[i : i + max(T_short, T_init_seq), :n_samples, :]
        train_data = train_data_shortened
        T = T_short
        n_samples = train_data_shortened.shape[1]

    print ('new number of samples', n_samples)

    if (args.model_type == 'ONET'):
        method = 1
    elif (args.model_type == 'HNET'):
        method = 5
    elif (args.model_type == 'RNN'):
        method = 3
    elif (args.model_type == 'LSTM'):
        method = 4
    else:
        raise ValueError('model_type not supported')

    if (args.leapfrog_train == 'true'):
        integrator_train = 'leapfrog'
    else:
        integrator_train = 'euler'
    if (args.leapfrog_test == 'true'):
        integrator_test = 'leapfrog'
    else:
        integrator_test = 'euler'


    log_dir = './logs/' + system_type + '_' + str(dataset_index) + str(run_index) + '/m' + str(method)
    pred_out_string = pred_dir + '/traj_pred_' + str(system_type) + '_' + str(method) + '_' + str(test_dataset_index) + '_' + str(run_index)

    if (not os.path.isdir(log_dir)):
        os.mkdir(log_dir)

    logger0 = Logger(os.path.join(log_dir, 'trloss.log'), print_out=True)
    logger1 = Logger(os.path.join(log_dir, 'teloss.log'), print_out=True)

    start = time.time()

    ## Training the model
    model, loss_record, val_loss_record = train(train_data, method=method, T=T, batch_size=batch_size, \
        n_epochs=n_epochs, n_samples=n_samples, n_val_samples=args.n_val_samples, dt=dt, lr=lr, n_layers=n_layers, \
        n_hidden=n_hidden, integrator_train=integrator_train, integrator_test=integrator_test, logger=logger0, device=device, \
        test_data=test_data, n_test_samples=n_test_samples, T_test=T_test, scheduler_type=scheduler_type, \
        scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor, pred_out_string=pred_out_string, \
        max_iters_init_train=max_iters_init_train, max_iters_init_test=max_iters_init_test, T_init_seq=T_init_seq, \
        coarsening_factor_test=coarsening_factor_test, test_freq=test_freq)


    f = open(os.path.join(log_dir, 'loss.pkl'), 'wb')
    # pickle.dump([np.array(loss_record), np.array(loss_record_val), t], f)
    pickle.dump([np.array(loss_record), np.array(val_loss_record)], f)
    f.close()
    loss_plot(os.path.join(log_dir, 'loss.pkl'), log_dir, name=['','','loss p','loss q'], teplotfreq = test_freq)
    loss_plot_restricted(os.path.join(log_dir, 'loss.pkl'), log_dir, name=['','','loss p','loss q'], teplotfreq = test_freq)

    train_time = time.time() - start
    print ('training with method ' + str(method) + ' costs time ', train_time)

    ## Predicting the test trajectories
    traj_pred = predict(test_data_init_seq=test_data[:T_init_seq, :, :], model=model, method=method, T_test=T_test, \
        n_test_samples=n_test_samples, dt=dt, integrator=integrator_test, device=device, max_iters_init=max_iters_init_test, \
        coarsening_factor=coarsening_factor_test)
    
    pred_time = time.time() - start
    print ('making the predictions with method ' + str(method) + ' costs time ', pred_time)

    np.save(pred_out_string + '.npy', traj_pred.cpu().data.numpy())
    torch.save(model.cpu(), model_dir + '/model_' + system_type + '_' + str(method) + '_' + str(dataset_index) + '_' + str(run_index))

    print ('done saving the predicted trajectory and trained model')


if __name__ == "__main__":
    main()
