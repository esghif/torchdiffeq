import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ode_demo import RunningAverageMeter


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class dXdt(nn.Module):

    def __init__(self):
        super(dXdt, self).__init__()

        self.layer = nn.Linear(1,1,bias=False)

    def forward(self, t, x):
        y = self.layer(t)
        return y


def test_f():

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    def dxdt(x, t):
        return 2*x

    def xt(t):
        return t**2

    f0 = np.array([0.0], dtype=np.float32)
    t = np.linspace(0, 1, 2, dtype=np.float32)
    t = np.expand_dims(t, axis=-1)

    f0 = torch.from_numpy(f0).to(device)
    t = torch.from_numpy(t).to(device)
    ft_true = xt(t)

    func = dXdt()
    func.layer.weight.data[0,0]=10.0
    print(func.layer.weight.data)
    func.to(device)

    optimizer = optim.SGD(func.parameters(), lr=1e-2)
    mse_loss = nn.MSELoss()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    ii = 0
    end = time.time()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        ft = odeint(func, f0, t)
        loss = mse_loss(ft, ft_true)
        loss.backward()

        #print(func.layer.weight.item(), func.layer.weight.grad.item())
        #print('dir(func.layer.weight.grad)', func.layer.weight.grad)
        #print('dir(func.layer.bias.grad)', func.layer.bias.grad)
        
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                ft = odeint(func, f0, t)
                loss = mse_loss(ft, ft_true)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                #visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()

    fig = plt.figure(figsize=(4, 4), facecolor='white')
    ax = fig.add_subplot(111, frameon=True)

    ft = odeint(func, f0, t)
    ft = ft.detach().cpu().numpy()
    t = t.detach().cpu().numpy()
    ft_true = ft_true.detach().cpu().numpy()

    ax.cla()
    ax.set_title('Trajectories')
    ax.set_xlabel('t')
    ax.set_ylabel('f')
    ax.plot(t, ft, 'b-')
    ax.plot(t, ft_true, 'r--')
    #ax.set_xlim(t.min(), t.max())
    #ax.set_ylim(-2, 2)
    #ax.legend()
    plt.show()


if __name__ == '__main__':
	test_f()
