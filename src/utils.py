# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Earlier versions of this file were written by Zhengdao Chen, used with permission.

import numpy as np
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os
import sys 
import argparse


class Logger():
    def __init__(self, f_path,print_out=True):
        self.path = f_path
        self.print_out = print_out
    def log(self,s):
        f = open(self.path,'a+')
        f.write(s+'\n')
        f.close()
        if self.print_out:
            print(s)

def loss_plot(in_path,out_dir,name=['pq sequence','image reconstruction', 'init pq', 'whole'], teplotfreq=1):
    f0 = open(in_path,'rb')
    # loss_tr,loss_te,bstidx = pickle.load(f0)
    loss_tr, loss_te = pickle.load(f0)
    f0.close()

    #fig = plt.figure((10,10))

    # for i in range(loss_tr.shape[0]):
    plt.figure(figsize=(10,5))
    plt.title('train loss')
    plt.xlabel('time step')
    plt.ylabel('mse loss')
    plt.plot(range(len(loss_tr)),loss_tr,label = 'train')
    plt.plot(range(0, len(loss_te) * teplotfreq, teplotfreq),loss_te,label = 'test')
    # plt.plot(range(len(loss_te)), loss_te[:, i], label='test')
    # plt.ylim(0, 1)
    plt.legend()
    plt.savefig(os.path.join(out_dir,'loss.png'))
    plt.close()


def loss_plot_restricted(in_path,out_dir,name=['pq sequence','image reconstruction', 'init pq', 'whole'], teplotfreq=1):
    f0 = open(in_path,'rb')
    loss_tr, loss_te = pickle.load(f0)
    f0.close()
    plt.figure(figsize=(10,5))
    plt.title('train loss')
    plt.xlabel('time step')
    plt.ylabel('mse loss')
    plt.plot(range(len(loss_tr)),loss_tr,label = 'train')
    plt.plot(range(0, len(loss_te) * teplotfreq, teplotfreq), loss_te,label = 'test')
    plt.ylim(0, 20)
    plt.legend()
    plt.savefig(os.path.join(out_dir,'loss_restricted.png'))
    plt.close()


def write_args(args,outdir,printout=True):
    dict_args = vars(args)
    f0 = open(os.path.join(outdir,'commend.txt'),'w')
    f1 = open(os.path.join(outdir,'args.txt'),'w')
    f0.write('python %s ' % sys.argv[0])

    for val in dict_args:
        f0.write(' --%s %s' % (val, str(dict_args[val])))
        s = '%20s %s' % (val,str(dict_args[val] if len(str(dict_args[val]))>0 else str('\'\'') ))
        f1.write(s+'\n')
        if printout:
            print(s)

    f0.close()
    f1.close()

