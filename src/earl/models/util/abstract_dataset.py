import pandas as pd
from tqdm import tqdm


class AbstractDataset:

    def __init__(self, env, bb_model, params):
        self.TARGET_NAME = 'Action'
        self.continuous_feature_names = []
        self.categorical_feature_names = []

        self.columns = self.generate_columns(params)

        self.env = env
        self.bb_model = bb_model
        self.params = params

        self.categorical_feature_names = self.params.categorical_features
        self.continuous_feature_names = self.params.continuous_features

        self.df = self.generate_dataset(env, bb_model)

        self.cat_order = {cat_feature: self.columns.index(cat_feature) for cat_feature in self.categorical_feature_names}

        self.state_shape = env.state_shape

    def get_dataset(self):
        return self.df

    def generate_dataset(self, env, bb_model):
        """
        Creates a csv dataset from given facts suitable for SL-based approaches
        """
        dataset = self.collect_dataset(env, bb_model)

        columns = self.columns+['Action']

        df = pd.DataFrame(dataset, columns=columns)

        return df

    def collect_dataset(self, env, bb_model):
        dataset = []
        n_ep = 10
        print('Generating dataset ...')
        for i in tqdm(range(n_ep)):
            obs, _ = env.reset()
            done = False
            while not done:
                action = bb_model.predict(obs)

                input = list(obs) + [action]

                dataset.append(input)

                obs, rew, done, trunc, info = env.step(action)

        print('Generated {} samples'.format(len(dataset)))
        return dataset

    def generate_columns(self, params):
        if 'columns' in params:
            return params.columns

        # TODO: generate template col names

    def transform_from_baseline_format(self, df):
        states = df[self.columns].values.tolist()

        return states

    def actionability_constraints(self):
        '''
        Returns a dictionary indicating the mutability of the features
        Used by the SGEN algorithm
        '''

        # TODO: extract from params or generate
        meta_action_data = {
            f'{f}': {'actionable': True,
                     'min': 0,
                     'max': 100,
                     'can_increase': True,
                     'can_decrease': True}
            for f in self.columns}

        return meta_action_data