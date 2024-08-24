import os
import logging

import torch
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class DQNModel:

    def __init__(self, env, model_path, params={}):
        self.model_path = model_path
        self.env = env
        self.params = params

        self.model = self.load_model(self.model_path, env)

    # TODO: load params same as the GANterfactual does

    def load_model(self, model_path, env):
        try:
            # try loading the model if already trained
            model = DQN.load(model_path)
            model.env = env
            model.policy.to('cpu')
            print('Loaded bb model')
        except FileNotFoundError:
            # train a new model
            print('Training bb model')
            n_env = Monitor(env, "./tensorboard/", allow_early_resets=True)
            n_env = DummyVecEnv([lambda: n_env])
            model = DQN('MlpPolicy',
                        n_env,
                        policy_kwargs=dict(net_arch=self.params['arch']),
                        learning_rate=self.params['lr'],
                        buffer_size=15000,
                        learning_starts=200,
                        batch_size=512,
                        gamma=self.params['gamma'],
                        train_freq=1,
                        gradient_steps=1,
                        target_update_interval=10,
                        exploration_fraction=self.params['exploration_fraction'],
                        verbose=1)

            model.learn(total_timesteps=self.params['training_timesteps'])
            model.save(model_path)

        return model

    def predict(self, x):
        ''' Predicts a deterministic action in state x '''
        action, _ = self.model.predict(x, deterministic=True)
        return action.item()

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