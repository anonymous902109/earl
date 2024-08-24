import ast
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def evaluate_explanations(env, params, eval_path, scenario, method_names, N_TEST):
    print('==================== Evaluating explanations ====================\n')
    evaluate_coverage(env, params, eval_path, scenario, method_names, N_TEST)
    evaluate_gen_time(env, params, eval_path, scenario, method_names)
    evaluate_properties(env, params, eval_path, scenario,method_names)
    evaluate_plausibility(env, params, eval_path, scenario, method_names)
    evaluate_diversity(env, params, eval_path, scenario, method_names)

def evaluate_coverage(env, params, eval_path, scenario, method_names, N_TEST):
    print('----------------- Evaluating coverage -----------------\n')
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Coverage (%)') + '|' + '\n'
    printout += '-' * ((20 + 1) * 2) + '\n'

    for method_name in method_names:
        method_dir = os.path.join(eval_path, params['task_name'], scenario, method_name)
        # for each outcome file
        unique_facts = 0.0
        outcomes = 0.0
        for outcome_file in os.listdir(method_dir):
            df_path = os.path.join(method_dir, outcome_file)
            df = pd.read_csv(df_path, header=0)

            unique_facts += len(df['Fact_id'].unique())
            outcomes += 1

        printout += '{: ^20}'.format(method_name) + '|' + \
                    '{: ^20.4}'.format(unique_facts / (N_TEST*outcomes) * 100) + '|' + '\n'

    print(printout + '\n')


def evaluate_gen_time(env, params, eval_path, scenario, method_names):
    print('----------------- Evaluating generation time -----------------\n')
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Generation Time (s)') + '|' + '\n'
    printout += '-' * ((20 + 1) * 2) + '\n'
    for method_name in method_names:
        method_dir = os.path.join(eval_path, params['task_name'], scenario, method_name)
        # for each outcome file
        times = []
        for outcome_file in os.listdir(method_dir):
            df_path = os.path.join(method_dir, outcome_file)
            df = pd.read_csv(df_path, header=0)

            times += list(df['gen_time'].values.squeeze())

        printout += '{: ^20}'.format(method_name) + '|' + '{: ^20.4}'.format(np.mean(times)) + '|' + '\n'

    print(printout + '\n')

def evaluate_properties(env, params, eval_path, scenario, method_names):
    print('----------------- Evaluating properties -----------------\n')
    print_header = True
    for method_name in method_names:
        method_dir = os.path.join(eval_path, params['task_name'], scenario, method_name)

        for outcome_file in os.listdir(method_dir):
            df_path = os.path.join(method_dir, outcome_file)
            df = pd.read_csv(df_path, header=0)

            properties = [col_name for col_name in df.columns if col_name not in
                          ['Fact_id', 'Found', 'Fact', 'Explanation', 'Recourse', 'Start State', 'Outcome State', 'True Actions']]

            if print_header == True:
                printout = '{: ^20}'.format('Algorithm') + '|' + \
                           ''.join(['{: ^20}'.format('{}'.format(p)) + '|' for p in properties]) + '\n'
                printout += '-' * ((20 + 1) * (len(properties))) + '\n'

                properties_dict = {p: [] for p in properties}

                print_header = False

            for p in properties:
                properties_dict[p] += list(df[p].values.squeeze())

        # scaling for feature-based properties before calculating averages
        printout += '{: ^20}'.format(method_name) + '|' + \
                    ''.join(['{: ^20.4}'.format(np.mean(list(properties_dict[p]))) + '|' for p in properties]) + '\n'

        # reset properties_dict for the next method
        properties_dict = {p: [] for p in properties}

    print(printout + '\n')


def evaluate_plausibility(env, params, eval_path, scenario, method_names):
    print('----------------- Evaluating plausibility -----------------\n')
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Plausible explanations (%)') + '|' + '\n'
    printout += '-' * ((20 + 1) * 2) + '\n'
    for method_name in method_names:
        method_dir = os.path.join(eval_path, params['task_name'], scenario,method_name)

        satisfied = 0.0
        total = 0.0
        # for each outcome file
        for outcome_file in os.listdir(method_dir):
            df_path = os.path.join(method_dir, outcome_file)
            df = pd.read_csv(df_path, header=0)
            if len(df) > 0:
                df['plausible'] = df.apply(lambda x: env.realistic(ast.literal_eval(x['Explanation'])), axis=1)
                satisfied += sum(df['plausible'])
                total += len(df)

        printout += '{: ^20}'.format(method_name) + '|' + \
                    '{: ^20.4}'.format(satisfied / (total) * 100) + '|' + '\n'

    print(printout + '\n')


def evaluate_diversity(env, params, eval_path, scenario,  method_names):
    print('----------------- Evaluating diversity -----------------\n')
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Number of explanations') + '|' + \
               '{: ^20}'.format('Feature Diversity') + '|' + \
               '{: ^20}'.format('Feature-based Metric Diversity') + '|' + \
               '{: ^20}'.format('RL Metric Diversity') + '|' + '\n'
    printout += '-' * ((20 + 1) * 5) + '\n'
    for method_name in method_names:
        method_dir = os.path.join(eval_path, params['task_name'], scenario, method_name)

        satisfied = 0.0
        total = 0.0
        # for each outcome file

        diversities = {'num_expl': [],
                       'feature_divs': [],
                       'feature_metric_divs': [],
                       'rl_metric_divs': []}

        for outcome_file in os.listdir(method_dir):
            df_path = os.path.join(method_dir, outcome_file)
            df = pd.read_csv(df_path, header=0)

            feature_metrics = ['Proximity', 'Sparsity']
            rl_metrics = [col for col in df.columns if
                          not col in ['Fact_id', 'Fact', 'Explanation', 'Recourse'] + feature_metrics]

            num_expl = evaluate_quantity(df)
            feature_div = evaluate_feature_diversity(df)
            feature_metrics_div = evaluate_metric_diversity(df, feature_metrics)
            rl_metrics_div = evaluate_metric_diversity(df, rl_metrics)

            diversities['num_expl'] += num_expl
            diversities['feature_divs'] += feature_div
            diversities['feature_metric_divs'] += feature_metrics_div
            diversities['rl_metric_divs'] += rl_metrics_div

        printout += '{: ^20}'.format(method_name) + '|' + \
                    '{: ^20.4}'.format(np.mean(diversities['num_expl'])) + '|' + \
                    '{: ^20.4}'.format(np.mean(diversities['feature_divs'])) + '|' + \
                    '{: ^20.4}'.format(np.mean(diversities['feature_metric_divs'])) + '|' + \
                    '{: ^20.4}'.format(np.mean(diversities['rl_metric_divs'])) + '|' + '\n'

    print(printout + '\n')

def evaluate_quantity(df):
    facts = pd.unique(df['Fact_id'])

    cfs = []
    for f in facts:
        n = len(df[df['Fact_id'] == f])
        cfs.append(n)

    return cfs

def evaluate_metric_diversity(df, metrics):
    facts = pd.unique(df['Fact_id'])
    diversity = []

    for f in facts:
        df_fact = df[df['Fact_id'] == f]
        for i, x in df_fact.iterrows():
            for j, y in df_fact.iterrows():
                if i != j:
                    diff = 0
                    for m in metrics:
                        diff += (x[m] - y[m]) ** 2

                    diversity.append(diff)

    return diversity


def evaluate_feature_diversity(df):
    facts = pd.unique(df['Fact_id'])
    diversity = []

    for f in facts:
        df_fact = df[df['Fact_id'] == f]
        for i, x in df_fact.iterrows():
            for j, y in df_fact.iterrows():
                if i != j:
                    diff = mse(np.array(ast.literal_eval(x['Explanation'])), np.array(ast.literal_eval(y['Explanation'])))
                    diversity.append(diff)

    return diversity


def mse(x, y):
    return np.sqrt(sum(np.square(x - y)))