from tqdm import tqdm

import pandas as pd

from src.earl.algorithms.s_gen.sgen_algorithm import SGENAlg
from src.earl.methods.abstract_method import AbstractMethod
from src.earl.models.util.abstract_dataset import AbstractDataset


class SGEN(AbstractMethod):
    ''' SGEN algorithm for generating semi-factual explanations '''
    def __init__(self, env, bb_model, params, diversity_size=3, pop_size=100, n_gen=10):
        self.diversity_size = diversity_size
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.bb_model = bb_model

        self.params = self.process_params(params)

        self.dataset = AbstractDataset(env, bb_model, self.params)

        self.alg = SGENAlg(self.bb_model,  self.diversity_size, self.pop_size, self.n_gen)

    def explain(self, fact, target):
        # append fact to the end of the dataset
        df = self.dataset.get_dataset()
        action = self.bb_model.predict(fact.state)
        fact_row = list(fact.state) + [tuple(action)]
        df = pd.concat([df, pd.DataFrame([fact_row], columns=df.columns)])

        # run SGEN for only one fact - the last row in the dataframe
        sf_df = self.alg.generate_sfs(self.dataset, df, target_action=action, test_ids=[len(df)-1])

        # transform generated explanation to the correct format
        sf_df['Explanation'] = self.dataset.transform_from_baseline_format(sf_df)

        return sf_df['Explanation'].values

    def generate_explanation(self, fact_dataset, fact_ids, target,  outcome):
        print('Running SGEN with diversity = {} for {} facts'.format(self.diversity_size, len(fact_ids)))
        df = fact_dataset.get_dataset()

        target_action = outcome.target_action
        facts_series = []
        fact_ids_series = []

        res = []
        for f in tqdm(fact_ids[0:1]):
            sf_df = self.alg.generate_sfs(fact_dataset, df, target_action, f)

            fact = fact_dataset.transform_from_baseline_format(df.iloc[f])
            for _, row in sf_df.iterrows():
                fact_ids_series.append(f)
                facts_series.append(list(fact))

            res.append(sf_df)

        res_df = pd.concat(res)
        res_df['Fact_id'] = fact_ids_series  # have to transform global to outcome-based index
        res_df['Fact'] = facts_series
        res_df['Explanation'] = fact_dataset.transform_from_baseline_format(res_df)

        res_df.drop([c for c in res_df.columns if c not in ['Fact_id', 'Explanation', 'Fact', 'gen_time']],
                    axis=1,
                    inplace=True)

        return res_df
