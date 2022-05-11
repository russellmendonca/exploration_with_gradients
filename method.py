import argparse
import gym
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

from wrapper_envs import HopperEnv, InvertedDoublePendulum

def identity(x):
    return x

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

class Mlp(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        hidden_activation=F.relu,
        output_activation=identity,
        b_init_value=0.0,
        init_w=1e-3,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            fanin_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size, bias=False)
        self.last_fc.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, take_hidden_output=False):
        h = x
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        if take_hidden_output:
            return h
        else:
            output = self.output_activation(self.last_fc(h))
            return output

def rollout(env, U):
    """
    Arguments:
        U: T x Action dim)
    Returns:
        X: (T x State dim)
        U: (T x Action dim)
        X_prime: (T x State dim)
    """
    o = env.reset()
    T = U.shape[0]
    X = np.zeros((T, o.shape[0]))
    X_prime = np.zeros_like(X)
    for i in range(T):
        X[i] = o
        o = env.step(U[i])[0]
        X_prime[i] = o
    return X, U, X_prime

def collect_data_and_assign_to_ensembles(env, num_ensembles, Us):
    """
    Arguments:
        Us: list of B (T x Action dim) arrays - where B is the number of trajectories
        num_ensembles:
    Returns:
        transitions: M (BxT)/M length arrays of transitions for each ensemble
    """
    Xs, X_primes = [], []
    for i in range(len(Us)):
        X, U, X_prime = rollout(env, Us[i])
        Xs.append(X)
        X_primes.append(X_prime)

    # concatenate into B*T length arrays
    Xs = np.concatenate(Xs)
    X_primes = np.concatenate(X_primes)
    Us = np.concatenate(Us)

    # randomly shuffle the entire array
    shuffled_indices = np.random.choice(list(range(Us.shape[0])), Us.shape[0:1], replace=False)
    Xs = Xs[shuffled_indices]
    X_primes = X_primes[shuffled_indices]
    Us = Us[shuffled_indices]

    # divide into even subarrays for each ensemble
    subarray_length = Us.shape[0] // num_ensembles
    transitions = [(Xs[i*subarray_length:(i+1)*subarray_length], Us[i*subarray_length:(i+1)*subarray_length], X_primes[i*subarray_length:(i+1)*subarray_length]) for i in range(num_ensembles)]
    return transitions

# X, U, X_prime
# Phi(X) + BU = X_prime -> train using MSE loss
# A = last layer weight of Phi
def fit_dynamics(transitions, phis, Bs, optimizers, iterations=10000, num_ensembles=1,):
    """
    #TODO: phi -> phis (list of nets for each ensemble)
    Arguments:
        transitions: M (BxT)/M length arrays of transitions for each ensemble
        phi: network that will map the state into a feature space
    Returns:
        As, Bs: lists of length M, A, B matrices for linear dynamics
    """
    batch_size = 32

    for i in range(num_ensembles):
        X, U, X_prime = transitions[i]
        phi = phis[i]
        B = Bs[i]
        optimizer = optimizers[i]
        valid_X, valid_U, valid_X_prime = transitions[i-1]
        for iteration in range(iterations):
            indices = np.random.choice(X.shape[0], size=(batch_size,))
            X_batch = X[indices]
            X_prime_batch = X_prime[indices]
            U_batch = U[indices]
            X_batch_phi = phi(torch.from_numpy(X_batch).float())
            X_prime_batch_phi = torch.from_numpy(X_prime_batch).float()
            optimizer.zero_grad()
            action_output = B(torch.from_numpy(U_batch).float())
            pred = X_batch_phi + action_output
            loss = F.mse_loss(pred, X_prime_batch_phi)
            loss.backward()
            optimizer.step()
            if (iteration+1) % 1000 == 0:
                print(f"Iteration: {iteration} train loss: {loss}")
                with torch.no_grad():
                    indices = np.random.choice(valid_X.shape[0], size=(batch_size,))
                    X_batch = valid_X[indices]
                    X_prime_batch = valid_X_prime[indices]
                    U_batch = valid_U[indices]
                    X_batch_phi = phi(torch.from_numpy(X_batch).float())
                    X_prime_batch_phi = torch.from_numpy(X_prime_batch).float()
                    action_output = B(torch.from_numpy(U_batch).float())
                    pred = X_batch_phi + action_output
                    loss = F.mse_loss(pred, X_prime_batch_phi)
                    print(f"Iteration: {iteration} valid loss: {loss}")

def compute_traj_fast(As, Bs, T, phis, tol=2e-1, max_iterations=1, x_rescaling=1, u_rescaling=3):
    M = len(As)
    n = len(As[0])
    m = len(Bs[0][0])
    Ac = As[-1]
    Bc = Bs[-1]
    Ae = As[:-1]
    Be = Bs[:-1]
    xs = Variable(torch.randn(T, n), requires_grad=True)
    us = Variable(torch.randn(T, m), requires_grad=True)
    lambdas = Variable(torch.randn(n * (T-1)), requires_grad=True)
    rho = 0.1

    def compute_obj():
        preds = torch.zeros((M-1, T, n))
        for m in range(M-1):
            preds[m] = phis[m](xs, take_hidden_output=True) @ Ae[m].T  + us @ Be[m].T
        mus = preds.mean(dim=0)
        objective = torch.pow(preds - mus, 2).sum()/((M-1) * T)
        return objective

    def compute_viol():
        return (phis[-1](xs[:-1, :], take_hidden_output=True) @ Ac.T + us[:-1, :] @ Bc.T - xs[1:, :]).reshape(-1)

    iters = 0
    lr = 1e-1
    best_u = None
    best_u_score = 0
    while iters < max_iterations:
        o = compute_obj()
        v = compute_viol()
        # print(f"Iters: {iters}, Obj: {o}, Cons:{torch.mean(torch.square(v))}")
        if torch.mean(torch.square(v)) < tol and o > best_u_score:
            best_u = us.detach().numpy()
            best_u_score = o
        # Primal Update
        L = o - lambdas.T @ v - rho * v.T @ v
        L.backward(retain_graph=True)
        H = -1 * rho * (Bs[-1].T @ Bs[-1]) * (1-2/M+1/M**2)
        for i in range(M-1):
            H = H + (Bs[i].T @ Bs[i]) * (1-2/(M-1)+1/(M-1)**2)
        H_inv = torch.linalg.inv(H)
        xs.data += lr * xs.grad.data
        xs.data = xs.data / torch.max(torch.abs(xs), dim=1)[0].reshape(-1, 1) * x_rescaling
        xs.grad = None
        us.data += us.grad.data @ H_inv.T
        us.data = us.data / torch.max(torch.abs(us), dim=1)[0].reshape(-1, 1) * u_rescaling
        us.grad = None
        # Dual Update
        if iters % 10 == 0:
            lambdas = lambdas + rho * v.T
            rho *= 1.5
        iters += 1
    print('Soln. Obj', best_u_score)
    if best_u is None:
        best_u = us.detach().numpy()
        print('failed to meet tolerance, taking latest us', torch.mean(torch.square(v)), o)
    return best_u

def load_array_and_plot(filename_random, filename_ours, plot_path, num_bins, title):
    random = np.load(filename_random)
    ours = np.load(filename_ours)
    coverage_random = np.where(random>0, 1, 0)
    coverage_random = (coverage_random/num_bins).sum(axis=-1)

    coverage_ours = np.where(ours>0, 1, 0)
    coverage_ours = (coverage_ours/num_bins).sum(axis=-1)

    plt.plot(coverage_random.mean(axis=0), label='random')
    plt.plot(coverage_ours.mean(axis=0), label='ours')

    plt.legend()
    plt.title(title)
    plt.savefig(plot_path)
    plt.clf()
