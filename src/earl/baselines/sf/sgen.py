from tqdm import tqdm

import pandas as pd

from src.earl.baselines.sf.algorithms.sgen_algorithm import SGENAlg


class SGEN():
    ''' SGEN algorithm for generating semi-factual explanations '''
    def __init__(self, bb_model, task_name='', diversity_size=3, pop_size=100, n_gen=10):
        self.diversity_size = diversity_size
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.bb_model = bb_model
        self.task_name = task_name

        self.alg = SGENAlg(self.bb_model,  self.diversity_size, self.pop_size, self.n_gen)

    def generate_explanation(self, fact_dataset, fact_ids, outcome):
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
