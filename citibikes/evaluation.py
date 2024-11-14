import ast
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def evaluate_explanations(env, eval_path, method_names, N_TEST):
    print('==================== Evaluating explanations ====================\n')
    # evaluate_coverage(env, eval_path, method_names, N_TEST)
    evaluate_gen_time(env, eval_path, method_names)
    evaluate_feature_similarity(env, eval_path, method_names)
    evaluate_plausibility(env, eval_path, method_names)
    evaluate_diversity(env, eval_path, method_names)

def evaluate_coverage(env, eval_path, method_names, N_TEST):
    print('----------------- Evaluating coverage -----------------\n')
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Coverage (%)') + '|' + '\n'
    printout += '-' * ((20 + 1) * 2) + '\n'

    for method_name in method_names:
        df_path = os.path.join(eval_path, '{}.csv'.format(method_name))
        df = pd.read_csv(df_path, header=0)

        unique_facts = len(df['Fact_id'].unique())

        printout += '{: ^20}'.format(method_name) + '|' + \
                    '{: ^20.4}'.format((unique_facts / N_TEST) * 100) + '|' + '\n'

    print(printout + '\n')


def evaluate_gen_time(env, eval_path, method_names):
    print('----------------- Evaluating generation time -----------------\n')
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Generation Time (s)') + '|' + '\n'
    printout += '-' * ((20 + 1) * 2) + '\n'
    for method_name in method_names:
        df_path = os.path.join(eval_path, '{}.csv'.format(method_name))
        df = pd.read_csv(df_path, header=0)

        times = list(df['gen_time'].values.squeeze())

        printout += '{: ^20}'.format(method_name) + '|' + '{: ^20.4}'.format(np.mean(times)) + '|' + '\n'

    print(printout + '\n')

def evaluate_feature_similarity(env, eval_path, method_names):
    print('----------------- Evaluating feature-similarity metrics -----------------\n')
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Proximity') + '|' + \
               '{: ^20}'.format('Sparsity') + '|'  + '\n'

    for method_name in method_names:
        df_path = os.path.join(eval_path, '{}.csv'.format(method_name))
        df = pd.read_csv(df_path, header=0)

        df['proximity'] = df.apply(lambda row: proximity(ast.literal_eval(row['fact']), ast.literal_eval(row['explanation'])), axis=1)
        df['proximity'] = df['proximity'] / max(df['proximity'])
        df['proximity'] = 1 - df['proximity']
        df['sparsity'] = df.apply(lambda row: sparsity(ast.literal_eval(row['fact']), ast.literal_eval(row['explanation'])), axis=1)

        printout += '{: ^20}'.format(method_name) + '|' + \
                    '{: ^20.4}'.format(np.mean(df['proximity'] / max(df['proximity']))) + '|' + \
                    '{: ^20.4}'.format(np.mean(df['sparsity'])) + '|' +'\n'

    print(printout)

def proximity(x, y):
    # MAXIMIZE
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm((x - y), ord=1)

def sparsity(x, y):
    # MINIMIZE
    return sum(np.array(x) != np.array(y)) / len(x)*1.0


def evaluate_properties(env, params, eval_path, scenario, method_names):
    print('----------------- Evaluating properties -----------------\n')
    print_header = True
    for method_name in method_names:
        df_path = os.path.join(eval_path, '{}.csv'.format(method_name))
        df = pd.read_csv(df_path, header=0)

        properties = ['Proximity', 'Sparsity']

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

    print(printout)

def evaluate_plausibility(env, eval_path, method_names):
    print('----------------- Evaluating plausibility -----------------\n')
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Plausible explanations (%)') + '|' + '\n'
    printout += '-' * ((20 + 1) * 2) + '\n'
    for method_name in method_names:

        df_path = os.path.join(eval_path, '{}.csv'.format(method_name))
        df = pd.read_csv(df_path, header=0)
        if len(df) > 0:
            df['plausible'] = df.apply(lambda x: env.realistic(ast.literal_eval(x['explanation'])), axis=1)
            satisfied = sum(df['plausible'])
            total = len(df)

        printout += '{: ^20}'.format(method_name) + '|' + \
                    '{: ^20.4}'.format(satisfied / (total) * 100) + '|' + '\n'

    print(printout + '\n')


def evaluate_diversity(env, params, eval_path, scenario,  method_names):
    print('----------------- Evaluating diversity -----------------\n')
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Number of explanations') + '|' + \
               '{: ^20}'.format('Feature Diversity') + '|'
    printout += '-' * ((20 + 1) * 3) + '\n'
    for method_name in method_names:
        df_path = os.path.join(eval_path, '{}.csv'.format(method_name))
        df = pd.read_csv(df_path, header=0)

        num_expl = evaluate_quantity(df)
        feature_div = evaluate_feature_diversity(df)

        printout += '{: ^20}'.format(method_name) + '|' + \
                    '{: ^20.4}'.format(np.mean(num_expl)) + '|' + \
                    '{: ^20.4}'.format(np.mean(feature_div)) + '|'

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