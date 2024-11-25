import ast
import os
from random import random

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.earl.evaluation.transformation import append_data, find_recourse


def evaluate_explanations(env, eval_path, method_names, N_TEST):
    print('==================== Evaluating explanations ====================\n')
    evaluate_coverage(env, eval_path, method_names, N_TEST)
    evaluate_gen_time(env, eval_path, method_names)
    evaluate_properties(env, eval_path, method_names)
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

        unique_facts = len(df['fact id'].unique())

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

def evaluate_properties(env, eval_path, method_names):
    print('----------------- Evaluating properties -----------------\n')
    properties = ['validity', 'sparsity', 'fidelity', 'exceptionality', 'uncertainty']

    printout = '{: ^20}'.format('Algorithm') + '|' + \
               ''.join(['{: ^20}'.format('{}'.format(p)) + '|' for p in properties]) + '\n'
    printout += '-' * ((20 + 1) * (len(properties))) + '\n'

    for method_name in method_names:
        df_path = os.path.join(eval_path, '{}.csv'.format(method_name))
        df = pd.read_csv(df_path, header=0)

        properties_dict = {p: [] for p in properties}

        for p in properties:
            try:
                properties_dict[p] += list(df[p].values)
            except KeyError:
                pass

        # scaling for feature-based properties before calculating averages
        printout += '{: ^20}'.format(method_name) + '|' + \
                    ''.join(['{: ^20.4}'.format(np.mean(list(properties_dict[p]))) + '|' for p in properties]) + '\n'

    print(printout + '\n')


def proximity(x, y):
    # MAXIMIZE
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm((x - y), ord=1)

def sparsity(x, y):
    # MINIMIZE
    return sum(np.array(x) != np.array(y)) / len(x)*1.0

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


def evaluate_diversity(env, eval_path,  method_names):
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
                    '{: ^20.4}'.format(np.mean(feature_div)) + '|' + '\n'

    print(printout + '\n')

def evaluate_quantity(df):
    facts = pd.unique(df['fact id'])

    cfs = []
    for f in facts:
        n = len(df[df['fact id'] == f])
        cfs.append(n)

    return cfs

def evaluate_metric_diversity(df, metrics):
    facts = pd.unique(df['fact id'])
    diversity = []

    for f in facts:
        df_fact = df[df['fact id'] == f]
        for i, x in df_fact.iterrows():
            for j, y in df_fact.iterrows():
                if i != j:
                    diff = 0
                    for m in metrics:
                        diff += (x[m] - y[m]) ** 2

                    diversity.append(diff)

    return diversity


def evaluate_feature_diversity(df):
    facts = pd.unique(df['fact id'])
    diversity = []

    for f in facts:
        df_fact = df[df['fact id'] == f]
        for i, x in df_fact.iterrows():
            for j, y in df_fact.iterrows():
                if i != j:
                    diff = mse(np.array(ast.literal_eval(x['explanation'])), np.array(ast.literal_eval(y['explanation'])))
                    diversity.append(diff)

    return diversity


def mse(x, y):
    return np.sqrt(sum(np.square(x - y)))

def transform_baseline_results(facts, objs, baseline_names, eval_path):
    ''' Finds the shortest path between the original instance and the counterfactual '''
     # for each baseline method
    for m_i, baseline_n in enumerate(baseline_names):
        baseline_path = os.path.join(eval_path, baseline_n)
        df = pd.read_csv(baseline_path, header=0)
        data = []

        for i, row in tqdm(df.iterrows()):
            fact_id = row['fact id']
            f = facts[fact_id]
            cf = row['explanation']

            solutions = []
            for o in objs:
                solutions += find_recourse(f, cf, o, {'gen_alg':
                                                          {'xu': [5, 5, 10],
                                                           'xl': [0, 0, 0],
                                                           'horizon': 5}})

            # no cfs found in the neighborhood
            if len(solutions) == 0:
                data = append_data(data,
                                   fact_id,
                                   list(f.forward_state),
                                   cf,
                                   None,
                                   [1 for i in objs[0].objectives + objs[0].constraints],
                                   row['gen_time'],
                                   0)
            else:
                 # select one at random if multiple ways to obtain the solution are present
                random_idx = random.choice(np.arange(0, len(solutions)))
                random_res = solutions[random_idx]

                # write down results
                data = append_data(data,
                                   fact_id,
                                   list(f.forward_state),
                                   cf,
                                   random_res.X,
                                   random_res.F + random_res.G,
                                   row['gen_time'], 1)

        columns = ['fact id',
                   'fact',
                   'explanation',
                   'recourse'] + objs[0].objectives + objs[0].constraints + ['gen time', 'found']

        df = pd.DataFrame(data, columns=columns)

        df.to_csv(baseline_n, index=False)
