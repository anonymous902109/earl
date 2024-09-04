import pandas as pd
from tqdm import tqdm

from src.earl.models.sl_fact import SLFact


class AbstractDataset:

    def __init__(self, env, bb_model, outcome, params):
        self.TARGET_NAME = 'Action'
        self.continuous_feature_names = []
        self.categorical_feature_names = []

        self.columns = self.generate_columns(params)

        self.env = env
        self.bb_model = bb_model
        self.outcome = outcome
        self.params = params
        self.target_action = outcome.target_action

        self.categorical_feature_names = self.params.categorical_features
        self.continuous_feature_names = self.params.continuous_features

        self.df = self.generate_dataset(env, bb_model, outcome)

        self.facts, self.fact_ids = self.get_facts(self.df, outcome)

        self.cat_order = {cat_feature: self.columns.index(cat_feature) for cat_feature in self.categorical_feature_names}

        self.state_shape = env.state_shape

    def get_dataset(self):
        return self.df

    def generate_dataset(self, env, bb_model, outcome):
        """
        Creates a csv dataset from given facts suitable for SL-based approaches
        """
        dataset = self.collect_dataset(env, bb_model, outcome)

        df = pd.DataFrame(dataset, columns=self.columns+['Outcome'])

        return df

    def collect_dataset(self, env, bb_model, outcome):
        dataset = []
        n_ep = 10
        print('Generating dataset ...')
        for i in tqdm(range(n_ep)):
            obs, _ = env.reset()
            done = False
            while not done:
                action = bb_model.predict(obs)

                explain_outcome = outcome.explain_outcome(env, obs)
                dataset.append(obs + [int(explain_outcome)])

                obs, rew, done, trunc, info = env.step(action)

        print('Generated {} samples'.format(len(dataset)))
        return dataset

    def get_facts(self, df, outcome):
        facts = []
        fact_ids = []

        for i, row in df.iterrows():
            if row['Outcome'] == 1:
                fact = SLFact(self.transform_from_baseline_format(row), outcome)

                facts.append(fact)
                fact_ids.append(i)

        return facts, fact_ids

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