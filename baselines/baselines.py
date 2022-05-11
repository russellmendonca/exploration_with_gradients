import sys
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from matplotlib import pyplot as plt
from utils import get_env_expldim_bins
import time

class BaselineEnv(gym.Env):
    def __init__(self, env_name, baseline_method=None, max_steps=100, disag_args = 1000):

    
        self.env , self.expl_dims, self.bins = get_env_expldim_bins(env_name)
        self.bin_counts = [np.zeros(len(self.bins[i])-1) for i in range(len(self.bins))]
        self.debug_bin_counts = [np.zeros(len(self.bins[i])-1) for i in range(len(self.bins))]
        self.max_steps = max_steps
        self.env._max_episode_steps = max_steps
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # self.grid_offset = -grid_max
        # self.grid_spacing = grid_spacing
        # state_count_shape = np.round(2*grid_max / grid_spacing).astype(int) + 1
        # self.state_count = np.zeros(state_count_shape)

        #forward model takes (x,th) pos and action, and predicts (x, th) of next state
        self.method = baseline_method
        if baseline_method == 'disag':
            _net_inp_space = self.observation_space.shape[0] + self.action_space.shape[0]
            _net_out_space = self.observation_space.shape[0]
            self.networks = [nn.Sequential(nn.Linear(_net_inp_space, 32), nn.ReLU(), nn.Linear(32,_net_out_space)) for _ in range(5)]
            _params = []
            for i in range(5):
                _params = _params + list(self.networks[i].parameters())

            self.optimizer = optim.Adam(_params, lr=1e-3)
        
            self.num_disag_train_steps = disag_args['num_train_steps']
            self.num_eps_for_disag_training  = disag_args['num_eps_for_training']


    def train_disag(self, obs_list, next_obs_list, act_list, train_all=True):
        #print(obs.shape, next_obs.shape, actions.shape)
        obs = np.concatenate(obs_list, axis = 0)
        next_obs = np.concatenate(next_obs_list, axis = 0)
        actions = np.concatenate(act_list, axis = 0)
        
        # if not train_all:
        #     sel = np.random.choice(np.arange(len(obs)), self.num_eps_for_disag_training*self.max_steps)
        #     obs = obs[sel]
        #     next_obs = next_obs[sel]
        #     actions  = actions[sel]
        
                   
        # _inp = torch.tensor(np.concatenate([obs, actions], axis = 1), dtype=torch.float32)
        for _ in range(self.num_disag_train_steps):
            total_loss = 0
            for net in self.networks:
                sel = np.random.choice(np.arange(len(obs)), 32)
                sel_obs = obs[sel]
                sel_next_obs = next_obs[sel]
                sel_acts  = actions[sel]
                _inp = torch.tensor(np.concatenate([sel_obs, sel_acts], axis = 1), dtype=torch.float32)

                self.optimizer.zero_grad()
                loss = torch.sum((net(_inp) - torch.tensor(sel_next_obs, dtype = torch.float32))**2)
                total_loss += loss
                loss.backward()
                self.optimizer.step()

        print('disag loss', loss)
    
    def get_disag_reward(self, obs, actions):
       
        _inp = torch.tensor(np.concatenate([obs, actions]), dtype=torch.float32)
        preds = torch.cat([torch.unsqueeze(net(_inp),1) for net in self.networks], dim = 1)
        #import ipdb ; ipdb.set_trace()
        return torch.mean(torch.std(preds, dim = 1), -1).numpy()
            #loss = torch.sum((self.forward_model(_inp) - torch.tensor(next_obs, dtype = torch.float32))**2)
        

    def step(self, action):
        # rew += self.beta / self.state_count[index]
        cur_obs = np.concatenate([self.env.sim.data.qpos, self.env.sim.data.qvel])
    
        obs, rew, og_done, info = self.env.step(action)
        if self.step_count ==  self.env._max_episode_steps-1:
            done = True
        else:
            done = False
        
        with torch.no_grad():
            if self.method == 'disag':
                rew = self.get_disag_reward(obs, action)
       
            elif self.method == 'count':
                rew = 0
                all_counts = []
                for i in range(len(self.bins)):
                    all_counts = []
                    idx = int((obs[i] - np.min(self.bins[i])) / 0.01)
                    if (-1 < idx) and (idx < len(self.bin_counts)) : 
                        all_counts.append(self.bin_counts[i][int(idx)])
                    elif idx <= -1:
                        all_counts.append(self.bin_counts[i][0])
                    else:
                        all_counts.append(self.bin_counts[i][-1])

                count = np.min(all_counts)
                rew = 1/(1+count)

        self.step_count+=1
        return obs, rew, done, info

    def reset(self):
        # self.state_count = np.zeros_like(self.state_count)
        self.step_count = 0
        return self.env.reset()

    def update_bin_counts_verify_consistent(self, obs):
         #very slow and inefficient
        for i in range(len(self.bins)):
            self.debug_bin_counts[i] += plt.hist(obs[:,i], self.bins[i])[0]

        for i in range(len(self.bins)):
            #idxs = np.clip((obs[:,i] - np.min(self.bins[i])) / 0.01, 0, len(self.bin_counts[i])-1).astype(np.int32)
            idxs = ((obs[:,i] - np.min(self.bins[i])) / 0.01).astype(np.int32)
            idxs = [_idx for _idx in idxs if (-1 < _idx and _idx < len(self.bin_counts[i])) ]
            for idx in idxs:
                self.bin_counts[i][idx] += 1
        
        for i in range(len(self.bins)):
            if not np.all(self.bin_counts[i] == self.debug_bin_counts[i]):
                import ipdb ; ipdb.set_trace()
                raise AssertionError('failed metric consistency test')

    def update_bin_counts(self, obs):
       
        for i in range(len(self.bins)):
            #idxs = np.clip((obs[:,i] - np.min(self.bins[i])) / 0.01, 0, len(self.bin_counts[i])-1).astype(np.int32)
            idxs = ((obs[:,i] - np.min(self.bins[i])) / 0.01).astype(np.int32)
            idxs = [_idx for _idx in idxs if (-1 < _idx and _idx < len(self.bin_counts[i])) ]
            for idx in idxs:
                self.bin_counts[i][idx] += 1
            
    def get_coverage(self):
        return [ np.count_nonzero(self.bin_counts[i])/len(self.bin_counts[i]) for i in range(len(self.bins))]

    def rollout(self, T, policy=None, rand_mode = False):
        obs = [self.reset()]
        actions = []
        rewards = []
        dones = []
        infos = []
        for t in range(T):
            if rand_mode or policy is None:
                action = self.action_space.sample()
            else:
                action, _ = policy.predict(obs[-1])
            ob, rew, done, info = self.step(action)
            obs.append(ob)
            actions.append(action)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
            if done:
                break
        return obs, rewards, dones, infos, actions



def get_max_min_env_limits(num_trajs = 1000, time_horizon = 100):
    
    for env_name in ['hopper', 'inverted_pendulum', 'inverted_double_pendulum']:
        env = BaselineEnv(env_name = env_name)
        
        obs_list = []
        for i in range(num_trajs):
            obs = env.rollout(T=time_horizon)[0]
            obs_list.append(np.array(obs))
        
        all_obs = np.concatenate(obs_list, axis = 0)
        print('max', np.max(all_obs, axis = 0))
        print('min', np.min(all_obs, axis = 0))




def run_exp(env_name, baseline_method, num_eps = 150, time_horizon = 100, num_prefill_eps = 10,
            disag_args = {'num_train_steps' : 20, 'num_eps_for_training': 20}):

    env = BaselineEnv(env_name = env_name, baseline_method = baseline_method, max_steps = time_horizon, disag_args = disag_args)
    model = PPO("MlpPolicy", env, verbose=1)
    obs_list = []
    act_list = []
    next_obs_list = []

    d1_coverage = []
    d2_coverage = []
    all_bins = [[] for i in range(len(env.expl_dims))]
    last_time = time.time()
    for ep in range(num_eps):
        obs, rews, dones, infos, actions = env.rollout(time_horizon, model, rand_mode = ep < num_prefill_eps)
        env.update_bin_counts(np.array(obs))
       
        obs_list.append(np.array(obs)[:-1, ])
        act_list.append(np.array(actions))
        next_obs_list.append(np.array(obs)[1:, ])
 
        if ep > num_prefill_eps:
            if env.method == 'disag':
                env.train_disag(obs_list, next_obs_list, act_list, train_all = ep < disag_args['num_eps_for_training'])

            model.learn(total_timesteps=time_horizon)
            for i, dim in enumerate(env.expl_dims):
                all_bins[i].append(env.bin_counts[dim].copy())
            print('num eps', ep, 'cov', env.get_coverage())
        print('ep time', time.time() - last_time)
        last_time = time.time()
    return [np.array(_bin) for _bin in  all_bins]
   


if __name__ == "__main__":

    method = 'disag'
    env_name = 'inverted_double_pendulum'
    num_seeds = 5
    _dir = 'logs_20trainsteps_bs32/' + method + '_' + env_name
    os.makedirs(_dir, exist_ok = True)
    all_bins = None
    for seed in range(num_seeds):
        all_bins = run_exp(env_name, method, num_eps = 550)
        for i in range(len(all_bins)):
            np.save(_dir + '/dim_'+str(i) +'_seed_'+str(seed), all_bins[i])