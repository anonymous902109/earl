import itertools
import os
import time
import shutil

import numpy as np
import pandas as pd
import torch

from src.earl.baselines.cf.algorithms.star_gan.dataset_generation import generate_dataset_gan
from src.earl.baselines.cf.algorithms.star_gan.model import Generator
from src.earl.baselines.cf.algorithms.star_gan.train import train_star_gan


class GANterfactual:
    def __init__(self, env, bb_model, dataset_size=int(5e5), num_features=10,
                 training_timesteps=int(5e3), batch_size=512, domains=None):
        self.env = env
        self.bb_model = bb_model
        self.dataset_size = dataset_size
        self.num_features = num_features
        self.training_timesteps = training_timesteps
        self.batch_size = batch_size

        # TODO: generator and discriminator architecture should be here too

        if domains is None:
            self.domains = self.generate_domains(self.env)
        else:
            self.domains = domains

        self.nb_domains = len(self.domains)

        self.model_save_path = os.path.join('trained_models', 'ganterfactual')

        self.generator_path = os.path.join(self.model_save_path, '{}-G.ckpt'.format(training_timesteps))

        try:
            self.generator = Generator(image_size=self.num_features, c_dim=self.nb_domains)
            self.generator.eval()
            self.generator.load_state_dict(torch.load(self.generator_path))
        except FileNotFoundError:
            self.run_ganterfactual()

    def generate_domains(self, env):
        action = env.action_space.sample()

        if isinstance(action, int):
            return np.arange(env.action_space.n)
        elif isinstance(action, np.ndarray) and len(action.shape) == 1:
            ns = [list(np.arange(0, env.action_space[i].n)) for i in range(len(env.action_space))]

            els = list(itertools.product(*ns))
            return els
        else:
            raise ValueError('Only Discrete and MultiDiscrete action spaces are supported')

    def run_ganterfactual(self):
        # TODO: params for ganterfactual should be in json format too
        # generate datasets for training ganterfactual if it does not exist already
        dataset_path = 'datasets/ganterfactual_data'
        if not os.path.isdir(os.path.join(dataset_path, 'test')):
            try:
                generate_dataset_gan(self.bb_model,
                                     self.env,
                                     dataset_path,
                                     self.dataset_size,
                                     self.nb_domains,
                                     self.domains)
            except (ValueError, TypeError) as err:
                print(err)
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)

        # train
        train_star_gan(image_size=self.num_features,
                       image_channels=1,
                       c_dim=self.nb_domains,
                       batch_size=self.batch_size,
                       domains=self.domains,
                       agent=self.bb_model,
                       num_iters=self.training_timesteps,
                       save_path=self.model_save_path,
                       dataset_path=dataset_path)

    def generate_explanation(self, fact_dataset, fact_ids, outcome, eval_path):
        result_data = []
        df = fact_dataset.get_dataset()

        target_action = outcome.target_action

        # select only indices which are in test_ids in the outcome section
        outcome_df = df[df['Outcome'] == target_action]
        outcome_df = outcome_df.iloc[:, :-1] # remove the last column which is outcome column
        test_ids = sorted(outcome_df.iloc[fact_ids].index.tolist())

        fact_id_series = []
        for i, x in enumerate(test_ids):
            fact_id = fact_ids[i]

            fact = outcome_df[outcome_df.index == x].values.squeeze()
            start = time.time()
            cf = self.get_best_cf(fact, target_action, fact_dataset)
            end = time.time()

            # check if explain is valid and only then include in the results
            if outcome.cf_outcome(np.array(cf).reshape(fact_dataset.state_shape)):
                fact_id_series.append(fact_id)
                result_data.append((x, fact, cf, end - start))

        columns = ['Fact_id', 'Fact', 'Explanation', 'gen_time']
        res_df = pd.DataFrame(result_data, columns=columns)
        res_df['Fact_id'] = fact_id_series

        res_df.to_csv(eval_path, index=False)

    def get_best_cf(self, fact, target, baseline_ds):
        ''' Returns all cfs found in the tree '''
        tensor_fact = torch.tensor(fact, dtype=torch.double).unsqueeze(0)
        cf = self.generate_counterfactual(tensor_fact, target, self.nb_domains)

        cf = cf.squeeze().tolist()

        # rounding for categorical features
        discrete_feature_ids = [i for i, c in enumerate(baseline_ds.columns) if c in baseline_ds.categorical_feature_names]
        cf = [round(feature) if i in discrete_feature_ids else feature for i, feature in enumerate(cf)]

        return cf

    def generate_counterfactual(self, fact, target, nb_domains):
        # convert target class to onehot
        onehot_target_class = np.zeros(nb_domains, dtype=int)
        onehot_target_class[target] = 1
        onehot_target_class = torch.tensor([onehot_target_class])

        # generate counterfactual
        counterfactual = self.generator.double()(fact, onehot_target_class)

        return counterfactual