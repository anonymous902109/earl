import os
import numpy as np


import logging

import torch
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class PPOModel:

    def __init__(self, env, model_path, params={}):
        self.model_path = model_path
        self.env = env
        self.params = params

        self.model = self.load_model(self.model_path, env)

    # TODO: load params same as the GANterfactual does

    def load_model(self, model_path, env):
        try:
            # try loading the model if already trained
            model = PPO.load(model_path)
            model.env = env
            model.policy.to('cpu')
            print('Loaded bb model')
        except FileNotFoundError:
            # train a new model
            print('Training bb model')
            n_env = Monitor(env, "./tensorboard/", allow_early_resets=True)
            n_env = DummyVecEnv([lambda: n_env])
            model = PPO('MlpPolicy', env, verbose=1,
                        policy_kwargs={'net_arch': [512, 512]},
                        learning_rate=0.0001,
                        batch_size=512,
                        gamma=0.9)

            model.learn(total_timesteps=self.params['training_timesteps'])
            model.save(model_path)

        return model

    def predict(self, x):
        ''' Predicts a deterministic action in state x '''
        action, _ = self.model.predict(x, deterministic=True)
        if len(action) == 1:
            return action.item()

        if isinstance(action, np.ndarray):
            return action.squeeze().tolist()

        # in case of multidim actions
        return action

    def predict_multiple(self, X):
        preds = []
        for state in X:
            preds.append(self.predict(state))

        return preds

    def get_action_prob(self, x, a):
        ''' Returns softmax probabilities of taking action a in x '''
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).unsqueeze(0)

        q_values = self.model.policy.q_net(x)
        probs = torch.softmax(q_values, dim=-1).squeeze()

        return probs[a].item()

    def predict_proba(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).unsqueeze(0)

        q_values = self.model.policy.q_net(x)
        probs = torch.softmax(q_values, dim=-1).squeeze()

        return probs

    def get_Q_vals(self, x):
        ''' Returns a list of Q values for taking any action in x '''
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).unsqueeze(0)

        q_values = self.model.policy.q_net(x)

        return q_values.squeeze().tolist()

    def evaluate(self):
        ''' Evaluates learned policy in the environment '''
        avg_rew = evaluate_policy(self.model, self.env, n_eval_episodes=10, deterministic=True)
        logging.info('Average reward = {}'.format(avg_rew))
        return avg_rew