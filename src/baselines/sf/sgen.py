from src.baselines.sf.algorithms.sgen_algorithm import SGENAlg


class SGEN():
    ''' SGEN algorithm for generating semi-factual explanations '''
    def __init__(self, bb_model, task_name='', diversity_size=3, params={}):
        self.diversity_size = diversity_size
        self.pop_size = 24
        self.bb_model = bb_model
        self.task_name = task_name

        self.alg = SGENAlg(self.bb_model,  self.diversity_size, self.pop_size)

    def generate_explanation(self, fact_dataset, fact_ids, outcome, path):
        df = fact_dataset.get_dataset()

        target_action = outcome.target_action

        # select only indices which are in test_ids in the outcome section
        outcome_df = df[df['Outcome'] == target_action]
        test_ids = sorted(outcome_df.iloc[fact_ids].index.tolist())

        sf_df = self.alg.generate_sfs(fact_dataset, df, target_action, test_ids)

        facts = fact_dataset.transform_from_baseline_format(df.iloc[test_ids])
        facts_series = []
        fact_ids_series = []
        for _, row in sf_df.iterrows():
            fact_id = fact_ids[list(test_ids).index(row['Fact_id'])]
            fact_ids_series.append(fact_id)
            fact = facts[list(test_ids).index(row['Fact_id'])]
            facts_series.append(list(fact))

        sf_df['Fact_id'] = fact_ids_series  # have to transform global to outcome-based index
        sf_df['Fact'] = facts_series
        sf_df['Explanation'] = fact_dataset.transform_from_baseline_format(sf_df)

        sf_df.drop([c for c in sf_df.columns if c not in ['Fact_id', 'Explanation', 'Fact', 'gen_time']], axis=1, inplace=True)

        sf_df.to_csv(path, index=False)

