import argparse

from src.models.abstract_dataset import AbstractDataset
from src.models.sl_fact import SLFact
from src.tasks.task import Task


class SLTask(Task):

    def __init__(self, env, bb_model, outcome, params={}):
        self.env = env
        self.bb_model = bb_model
        self.outcome = outcome
        self.params = params

        self.params = self.parse_arguments(params)

        self.dataset = AbstractDataset(env, bb_model, outcome, self.params)
        self.facts, self.fact_ids = self.get_facts(self.dataset)

    def get_facts(self, dataset):
        facts = []
        fact_ids = []
        for i, row in dataset.df.iterrows():
            if row['Outcome'] == 1:
                state = dataset.transform_from_baseline_format(row)
                f = SLFact(state, self.outcome)

                facts.append(f)
                fact_ids.append(i)

        return facts, fact_ids

    def sample_facts(self):
        pass

    def explain(self, algorithm):
        sfs = algorithm.generate_explanation(self.dataset, self.fact_ids, self.outcome)

        return sfs

    def parse_arguments(self, params):
        def list_of_strings(arg):
            import ast
            return ast.literal_eval(arg)

        parser = argparse.ArgumentParser()
        parser.add_argument('--columns', type=list_of_strings, default=[], help='a list of column names for state features')
        parser.add_argument('--categorical_features', type=list_of_strings, default=[], help='a list of categorical state features')
        parser.add_argument('--continuous_features', type=list_of_strings, default=[], help='a list of continuous state features')

        args = parser.parse_args(params)

        return args
