import copy
import inspect

import gymnasium as gym
import highway_env
from src.envs.abs_env import AbstractEnv
import numpy as np


class HighwayEnv(AbstractEnv):

    def __init__(self):
        ''' Wrapper for the highway gym environment containing additional methods need for running counterfactual methods '''

        # define gym environment
        self.gym_env = gym.make("highway-fast-v0", render_mode="rgb_array")
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                "grid_step": [5, 5],
                "absolute": False
            }
        }
        self.gym_env.configure(config)
        self.gym_env.reset()

        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space

        # define action space as discrete or cont
        self.action_type = 'discrete'

        self.render_mode = 'rgb_array'

    def step(self, action):
        ''' Runs a step() action in the environment compatible with Gym framework'''
        obs, rew, done, trunc, info = self.gym_env.step(action)

        # define failure (in highway it's the collision with another vehicle)
        self.failure = info['rewards']['collision_reward'] > 0

        self.state = np.array(obs)

        if done or trunc:
            self.is_done = True

        return self.state, rew, done, trunc, info

    def reset(self, seed=0, options={}):
        ''' Runs reset() action in the environment compatible with Gym framework'''
        self.is_done = False

        self.failure = False  # reset failure
        obs, info = self.gym_env.reset(seed=seed)

        self.state = np.array(obs)

        return self.state, info

    def render(self):
        return self.gym_env.render()

    def get_actions(self, x):
        """ Returns a list of actions available in state x"""
        return np.arange(0, self.action_space.n)

    def set_nonstoch_state(self, state, env_state):
        '''
        Copies state information from x to the current environment without setting the random generator.
        In highway environment, this means all cars are in the same position as in x, but their behavior might be stochastic
        compared to x.
        Copies road information about vehicles but not the random generator to ensure that different paths
        can be taken according to stochasticity '''
        for attribute in [a for a in dir(env_state) if not a.startswith('__')]:
            if attribute != 'unwrapped' and not inspect.ismethod(getattr(self.gym_env.unwrapped, attribute)) and attribute != 'np_random' and attribute != '_np_random':
                if attribute == 'road':
                    for road_attr in [a for a in dir(env_state.road)]:
                        if road_attr != 'np_random' and not road_attr.startswith('__'):
                            self.gym_env.unwrapped.road.__setattr__(attribute, copy.copy(getattr(env_state.road, road_attr)))

                else:
                    self.gym_env.unwrapped.__setattr__(attribute, copy.copy(getattr(env_state, attribute)))

    def set_stochastic_state(self, state, env_state):
        ''' Copies the whole state including the random generator so that the path taken from the state is deterministic '''
        for attribute in [a for a in dir(env_state) if not a.startswith('__')]:
            if attribute != 'unwrapped':
                self.gym_env.unwrapped.__setattr__(attribute, copy.copy(getattr(env_state, attribute)))

    def check_done(self, x):
        """ Returns a boolean indicating if x is a terminal state in the environment"""
        return False

    def equal_states(self, x1, x2):
        """ Returns a boolean indicating if x1 and x2 are the same state"""
        pass

    def writable_state(self, x):
        """ Returns a string with all state information to be used for writing results"""
        pass

    def check_failure(self):
        ''' Returns a boolean indicating if a failure occured in the environment'''
        return self.failure

    def get_env_state(self):
        return self.gym_env.unwrapped

    def action_distance(self, a, b):
        return a != b





