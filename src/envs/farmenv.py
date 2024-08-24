import copy
from datetime import datetime

from farm_games_local.farmgym_games.game_builder.utils_sb3 import farmgym_to_gym_observations_flattened, wrapper
from src.envs.abs_env import AbstractEnv
import numpy as np
import gymnasium as gym

from farm_games_local.farmgym_games.game_catalogue.farm0.farm import env as Farm0


class FarmEnv(AbstractEnv):

    def __init__(self):
        self.gym_env = Farm0()
        self.gym_env.reset()

        self.gym_env.farmgym_to_gym_observations = farmgym_to_gym_observations_flattened
        self.gym_env = wrapper(self.gym_env)

        obs, _ = self.gym_env.reset()

        self.state_dim = len(obs)
        self.action_dim = self.gym_env.action_space.n
        self.observation_space = gym.spaces.Box(low=np.zeros((self.state_dim, )), high=np.array([1000] * self.state_dim), shape=(self.state_dim, ))
        self.action_space = gym.spaces.Discrete(self.action_dim)

        self.action_type = 'cont'
        self.action_space_low = 0
        self.action_space_high = self.gym_env.action_space.n


    def step(self, action):
        obs, rew, done, trunc, _ = self.gym_env.step(action)

        self.state = np.array(obs)

        if done or trunc:
            self.is_done = True

        return self.state, rew, done, trunc,  {"composite_obs": obs}

    def reset(self, seed=None):
        self.is_done = False
        if seed is None:
            seed = int(datetime.now().timestamp()*10000)

        obs, info = self.gym_env.reset(seed)

        self.state = np.array(obs)

        return self.state, info

    def render(self):
        print(self.state)

    # def flatten_obs(self, obs):
    #     # take only the first element of the tuple that contains the obs -- only after reset
    #     obs = copy.copy(obs)
    #     if isinstance(obs, tuple):
    #         obs = obs[0]
    #
    #     flat_obs = []
    #     # obs is a list of dictionaries
    #     for d in obs:
    #         while isinstance(d, dict):
    #             vals = list(d.values()) # there is only one key-value pair in all dicts
    #             d = vals[0]
    #
    #         if isinstance(vals, list):
    #             for e in vals:
    #                 flat_obs.append(e)
    #         else:
    #             flat_obs.append(vals)
    #
    #     return [e if not isinstance(e, list) else e[0] for e in flat_obs]  # flatten list

    def decode_action(self, action):
        return [action]

    def get_actions(self, x):
        """ Returns a list of actions available in state x"""
        return np.arange(0, self.action_space.n)

    def set_stochastic_state(self, state, env_state):
        self.set_state(copy.deepcopy(env_state))
        self.gym_env.np_random = env_state[0]

    def set_nonstoch_state(self, state, env_state):
        self.set_state(copy.deepcopy(env_state))

        # reset random generators to allow stochasticity
        self.gym_env.np_random = np.random.RandomState(int(datetime.now().timestamp()))
        for e in self.gym_env.fields['Field-0'].entities:
            self.gym_env.fields['Field-0'].entities[e].np_random = np.random.RandomState(int(datetime.now().timestamp()))

    def set_state(self, x):
        # TODO: do this for all params not just ones passed by obs
        """ Changes the environment"s current state to x """
        field = self.gym_env.fields['Field-0']

        field.entities['Weather-0'] = x[1]['Weather-0']
        field.entities['Soil-0'] = x[1]['Soil-0']
        field.entities['Plant-0'] = x[1]['Plant-0']

        self.gym_env.farmers['BasicFarmer-0'].fields['Field-0'] = field

    def check_done(self, x):
        """ Returns a boolean indicating if x is a terminal state in the environment"""
        return False

    def equal_states(self, x1, x2):
        """ Returns a boolean indicating if x1 and x2 are the same state"""
        return sum(x1 != x2) == 0

    def writable_state(self, x):
        """ Returns a string with all state information to be used for writing results"""
        return [list(x)]

    # def unflatten_obs(self, x):
    #
    #     x = {'day#int365': x[0],
    #          'max#°C': x[1],
    #          'mean#°C': x[2],
    #          'min#°C': x[3],
    #          'consecutive_dry#day': x[4],
    #          'stage': x[5],
    #          'population#nb': x[6],
    #          'size#cm': x[7],
    #          'fruits_per_plant#nb': x[8],
    #          'fruit_weight#g': x[9]}
    #
    #     return x

    def check_failure(self):
        ''' Returns a boolean indicating if a failure occured in the environment'''
        return self.failure

    def get_env_state(self):
        return (self.gym_env.np_random, self.gym_env.fields['Field-0'].entities)


