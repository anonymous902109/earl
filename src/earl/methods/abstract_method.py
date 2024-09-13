import argparse


class AbstractMethod():

    def __init__(self):
        pass

    def explain(self, fact, target):
        pass

    def process_params(self, params):
        def list_of_strings(arg):
            import ast
            return ast.literal_eval(arg)

        parser = argparse.ArgumentParser()
        parser.add_argument('--columns', type=list_of_strings, default=[],
                            help='a list of column names for state features')
        parser.add_argument('--categorical_features', type=list_of_strings, default=[],
                            help='a list of categorical state features')
        parser.add_argument('--continuous_features', type=list_of_strings, default=[],
                            help='a list of continuous state features')

        args = parser.parse_args(params)

        return args