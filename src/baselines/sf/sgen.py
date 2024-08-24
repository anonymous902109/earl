from src.baselines.sf.algorithms.sgen_algorithm import SGENAlg
import pandas as pd


class SGEN():
    ''' SGEN algorithm for generating semi-factual explanations '''
    def __init__(self, bb_model, task_name='', diversity_size=3, params={}):
        self.diversity_size = diversity_size
        self.pop_size = 24
        self.bb_model = bb_model
        self.task_name = task_name

        self.alg = SGENAlg(self.bb_model,  self.diversity_size, self.pop_size)

    def generate_explanation(self, fact_dataset, fact_ids, outcome):
        df = fact_dataset.get_dataset()

        target_action = outcome.target_action
        facts_series = []
        fact_ids_series = []

        res_df = pd.DataFrame()
        for f in fact_ids:
            sf_df = self.alg.generate_sfs(fact_dataset, df, target_action, f)

            fact = fact_dataset.transform_from_baseline_format(df.iloc[f])
            for _, row in sf_df.iterrows():
                fact_ids_series.append(f)
                facts_series.append(list(fact))

            res_df = pd.concat([res_df, sf_df])

        res_df['Fact_id'] = fact_ids_series  # have to transform global to outcome-based index
        res_df['Fact'] = facts_series
        sf_df['Explanation'] = fact_dataset.transform_from_baseline_format(res_df)

        res_df.drop([c for c in sf_df.columns if c not in ['Fact_id', 'Explanation', 'Fact', 'gen_time']], axis=1, inplace=True)

        # sf_df.to_csv(path, index=False)
        return res_df
