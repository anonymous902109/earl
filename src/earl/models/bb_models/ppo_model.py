import os
import numpy as np


import logging

import torch
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class PPOModel:

    def __init__(self, env, model_path,
                 arch=[128, 128], training_timesteps=int(1e5), lr=1e-3,
                 batch_size=128, gamma=0.9,  verbose=0):

        self.model_path = model_path
        self.env = env

        self.arch = arch
        self.training_timesteps = training_timesteps
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose

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
            model = PPO('MlpPolicy',
                        env,
                        verbose=self.verbose,
                        policy_kwargs={'net_arch': self.arch},
                        learning_rate=self.lr,
                        batch_size=self.batch_size,
                        gamma=self.gamma)

            model.learn(total_timesteps=self.training_timesteps)
            model.save(model_path)

        return model

    def predict(self, x):
        ''' Predicts a deterministic action in state x '''
        action, _ = self.model.predict(x, deterministic=True)
        if isinstance(action, np.ndarray):
            return action.squeeze().tolist()
        elif len(action) == 1:
            return action.item()

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

        if not isinstance(a, list) and not isinstance(a, tuple):
            a = [a]

        prob = 0.0
        distribution = self.model.policy.get_distribution(x.squeeze().reshape(1, -1)).distribution

        for i, action_component in enumerate(distribution):
            prob += action_component.probs.squeeze()[a[i]]

        return prob.item()

    def get_importance(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).unsqueeze(0)

        distribution = self.model.policy.get_distribution(x.squeeze().reshape(1, -1)).distribution
        imp = 0.0

        for i, action_component in enumerate(distribution):
            q_vals = action_component.probs.squeeze()
            imp += abs(max(q_vals) - min(q_vals))

        return imp.item()

    def evaluate(self):
        ''' Evaluates learned policy in the environment '''
        avg_rew = evaluate_policy(self.model, self.env, n_eval_episodes=10, deterministic=True)
        return avg_rew