
import gym
import numpy as np

from gym.spaces import Box

class AntEnv():
    def __init__(self, max_steps = 100):
        self.env = gym.make("Ant-v2")
        self.env._max_episode_steps = max_steps

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self._get_obs()
        return obs, rew, done, info

    def reset_model(self):
        _ = self.env.reset_model()
        return self._get_obs()


    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

class HopperEnv():
    def __init__(self, max_steps = 100):
        self.env = gym.make("Hopper-v2")
        self.env._max_episode_steps = max_steps
        self.observation_space = Box(-np.ones(12), np.ones(12))

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self._get_obs()
        return obs, rew, done, info

    def reset(self):
        self.env.reset()
        return self._get_obs()


    def _get_obs(self):
        return np.concatenate(
            [self.sim.data.qpos.flat, np.clip(self.sim.data.qvel.flat, -10, 10)]
        )

class InvertedDoublePendulum():
    def __init__(self, max_steps = 100):
        self.env = gym.make("InvertedDoublePendulum-v2")
        self.env._max_episode_steps = max_steps
        self.observation_space = Box(np.ones(self._get_obs().shape)*-np.inf, np.ones(self._get_obs().shape)*-np.inf)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_joint_angles(self):
        return self.sim.data.qpos[1:]

    def reset(self):
        self.env.reset()
        return self._get_obs()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self._get_obs()
        info['joint_angles'] = self.get_joint_angles()
        return obs, rew, done, info

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qpos[1] = qpos[1] % (2 * np.pi)
        qpos[2] = qpos[2] % (2 * np.pi)
        return np.concatenate(
            [
                qpos,
                np.clip(self.sim.data.qvel, -10, 10),
                np.clip(self.sim.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

if __name__ == '__main__':
    # env = HopperEnv()
    env = InvertedDoublePendulum()
    env.reset()
    for i in range(1000):
        action = env.action_space.sample()

        obs, rew, done, info = env.step(action)
        print(obs)
